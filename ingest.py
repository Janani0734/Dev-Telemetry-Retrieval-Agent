"""
Dev Telemetry Retrieval Agent - Qdrant Cloud hydration script.

Pulls the official LangGraph Python documentation index (llms.txt), fetches each
page as markdown, chunks it, embeds it at 768 dimensions and upserts it into the
'langgraph_docs' collection.

Usage:
    python ingest.py                 # first 10 pages
    python ingest.py --limit 43      # everything in the index
    python ingest.py --recreate      # drop and rebuild the collection first
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
import uuid
from functools import lru_cache
from typing import Any, Iterable

import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("dev_telemetry.ingest")

COLLECTION_NAME = "langgraph_docs"
VECTOR_SIZE = 768
EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# The docs are served from docs.langchain.com, not langchain.com.
INDEX_URL = "https://docs.langchain.com/oss/python/langgraph/llms.txt"

DEFAULT_PAGE_LIMIT = 10
EMBED_BATCH_SIZE = 32
HTTP_TIMEOUT = 20
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 5

MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")


def resolve_google_api_key() -> str | None:
    """langchain-google-genai reads GOOGLE_API_KEY; accept GEMINI_API_KEY too."""
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


_api_key = resolve_google_api_key()
if _api_key:
    os.environ.setdefault("GOOGLE_API_KEY", _api_key)


# --------------------------------------------------------------------------- #
# Cached clients - same helper names as app.py and engine.py
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=60,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    gemini-embedding-001 emits 3072 dimensions by default; the collection is
    768-dim, so the output size is pinned to keep upserts valid.
    """
    kwargs: dict[str, Any] = {
        "model": EMBED_MODEL,
        "task_type": "retrieval_document",
        "google_api_key": resolve_google_api_key(),
    }
    try:
        return GoogleGenerativeAIEmbeddings(output_dimensionality=VECTOR_SIZE, **kwargs)
    except TypeError:
        LOGGER.warning(
            "Installed langchain-google-genai does not expose output_dimensionality; "
            "upserts into a %s-dim collection will be rejected.", VECTOR_SIZE
        )
        return GoogleGenerativeAIEmbeddings(**kwargs)


@lru_cache(maxsize=1)
def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    exists = client.collection_exists(COLLECTION_NAME)
    if exists and recreate:
        LOGGER.warning("Dropping existing collection '%s'.", COLLECTION_NAME)
        client.delete_collection(COLLECTION_NAME)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        LOGGER.info("Created collection '%s' (%s-dim, cosine).", COLLECTION_NAME, VECTOR_SIZE)
    else:
        LOGGER.info("Reusing existing collection '%s'.", COLLECTION_NAME)


# --------------------------------------------------------------------------- #
# Index parsing
# --------------------------------------------------------------------------- #

def fetch_index(url: str = INDEX_URL) -> list[tuple[str, str]]:
    """
    llms.txt is a markdown link list: '- [Title](https://.../page.md)'.
    Returns [(title, url), ...]; falls back to bare http lines if the format
    ever changes.
    """
    try:
        response = requests.get(url, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        LOGGER.error("Index fetch returned HTTP %s for %s", status, url)
        return []
    except requests.exceptions.RequestException as exc:
        LOGGER.error("Could not reach the documentation index (%s): %s", url, exc)
        return []

    entries = [(title.strip(), href.strip()) for title, href in MARKDOWN_LINK.findall(response.text)]
    if not entries:
        LOGGER.warning("No markdown links found; falling back to bare URL lines.")
        entries = [
            (line.strip(), line.strip())
            for line in response.text.splitlines()
            if line.strip().startswith("http")
        ]

    # Keep markdown source pages and drop duplicates while preserving order.
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for title, href in entries:
        if href in seen:
            continue
        seen.add(href)
        deduped.append((title, href))

    LOGGER.info("Parsed %s document entries from the index.", len(deduped))
    return deduped


def fetch_page(url: str) -> str | None:
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(url, timeout=HTTP_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status in (429, 500, 502, 503, 504) and attempt < RETRY_ATTEMPTS:
                wait = RETRY_BACKOFF_SECONDS * attempt
                LOGGER.warning("HTTP %s on %s - retrying in %ss.", status, url, wait)
                time.sleep(wait)
                continue
            LOGGER.error("Giving up on %s (HTTP %s).", url, status)
            return None
        except requests.exceptions.RequestException as exc:
            if attempt < RETRY_ATTEMPTS:
                LOGGER.warning("Transport error on %s (%s) - retrying.", url, exc)
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
                continue
            LOGGER.error("Transport failure on %s: %s", url, exc)
            return None
    return None


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start:start + size]


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #

def embed_with_retry(chunks: list[str]) -> list[list[float]] | None:
    embeddings = get_embeddings()
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return embeddings.embed_documents(chunks)
        except Exception as exc:
            if attempt < RETRY_ATTEMPTS:
                wait = RETRY_BACKOFF_SECONDS * attempt
                LOGGER.warning(
                    "Embedding batch failed (%s: %s) - retrying in %ss.",
                    type(exc).__name__, exc, wait,
                )
                time.sleep(wait)
                continue
            LOGGER.error("Embedding batch failed permanently (%s: %s)", type(exc).__name__, exc)
            return None
    return None


def ingest_docs(limit: int = DEFAULT_PAGE_LIMIT, recreate: bool = False) -> int:
    if not resolve_google_api_key():
        LOGGER.error("Set GEMINI_API_KEY (or GOOGLE_API_KEY) before running ingestion.")
        return 0
    if not os.environ.get("QDRANT_URL"):
        LOGGER.error("Set QDRANT_URL before running ingestion.")
        return 0

    try:
        client = get_qdrant_client()
        ensure_collection(client, recreate=recreate)
    except (UnexpectedResponse, ApiException) as exc:
        LOGGER.error("Qdrant rejected the collection setup: %s", exc)
        return 0
    except Exception as exc:
        LOGGER.error("Could not connect to Qdrant (%s: %s)", type(exc).__name__, exc)
        return 0

    entries = fetch_index()
    if not entries:
        LOGGER.error("Index is empty - nothing to ingest.")
        return 0

    targets = entries[:limit]
    LOGGER.info("Hydrating %s of %s pages.", len(targets), len(entries))

    splitter = get_splitter()
    total_points = 0

    for idx, (title, url) in enumerate(targets, start=1):
        body = fetch_page(url)
        if not body:
            LOGGER.warning("[%s/%s] Skipped %s", idx, len(targets), url)
            continue

        chunks = [c for c in splitter.split_text(body) if c.strip()]
        if not chunks:
            LOGGER.warning("[%s/%s] No text blocks produced from %s", idx, len(targets), url)
            continue

        page_points = 0
        chunk_offset = 0
        for batch in batched(chunks, EMBED_BATCH_SIZE):
            vectors = embed_with_retry(batch)
            if vectors is None:
                LOGGER.error("[%s/%s] Aborting page after embedding failure.", idx, len(targets))
                break

            if len(vectors[0]) != VECTOR_SIZE:
                LOGGER.error(
                    "Dimension mismatch: model returned %s-dim vectors, collection expects %s. "
                    "Upgrade langchain-google-genai or recreate the collection at that size.",
                    len(vectors[0]), VECTOR_SIZE,
                )
                return total_points

            points = [
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}_{chunk_offset + i}")),
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source_url": url,
                        "section": title,
                    },
                )
                for i, (chunk, vector) in enumerate(zip(batch, vectors))
            ]
            chunk_offset += len(batch)

            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
                page_points += len(points)
            except (UnexpectedResponse, ApiException) as exc:
                LOGGER.error("Qdrant rejected an upsert batch for %s: %s", url, exc)
                break
            except Exception as exc:
                LOGGER.error("Upsert failed for %s (%s: %s)", url, type(exc).__name__, exc)
                break

        total_points += page_points
        LOGGER.info("[%s/%s] %s - %s chunks indexed.", idx, len(targets), title, page_points)

    try:
        count = client.count(COLLECTION_NAME, exact=True).count
        LOGGER.info("Hydration complete. Collection now holds %s points.", count)
    except Exception as exc:
        LOGGER.warning("Could not read final collection count (%s: %s)", type(exc).__name__, exc)

    return total_points


def main() -> int:
    parser = argparse.ArgumentParser(description="Hydrate the langgraph_docs Qdrant collection.")
    parser.add_argument(
        "--limit", type=int, default=DEFAULT_PAGE_LIMIT,
        help=f"Number of documentation pages to ingest (default {DEFAULT_PAGE_LIMIT}).",
    )
    parser.add_argument(
        "--recreate", action="store_true",
        help="Drop and rebuild the collection before ingesting.",
    )
    args = parser.parse_args()

    written = ingest_docs(limit=args.limit, recreate=args.recreate)
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())

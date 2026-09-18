"""
Dev Telemetry Retrieval Agent - LangGraph engine.

Routes a developer query to either the documentation RAG pipeline (Qdrant Cloud)
or the live GitHub Issues search API, then synthesizes a cited answer.

Shares its configuration constants and client helpers with app.py and ingest.py.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Annotated, Any, Literal, Sequence, TypedDict

import requests
from langchain_core.messages import BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException, UnexpectedResponse

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger("dev_telemetry.engine")

COLLECTION_NAME = "langgraph_docs"
VECTOR_SIZE = 768                      # must match the Qdrant collection schema
EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-2.5-flash"

TOP_K_DOCS = 4
MAX_ISSUES = 3

GITHUB_SEARCH_URL = "https://api.github.com/search/issues"
GITHUB_REPO = "langchain-ai/langgraph"
GITHUB_QUERY_MAX_CHARS = 200           # GitHub rejects q strings over 256 chars
HTTP_TIMEOUT = 15

TROUBLESHOOTING_KEYWORDS = (
    "error", "exception", "traceback", "fail", "looping", "stuck",
)


def resolve_google_api_key() -> str | None:
    """langchain-google-genai reads GOOGLE_API_KEY; accept GEMINI_API_KEY too."""
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


_api_key = resolve_google_api_key()
if _api_key:
    os.environ.setdefault("GOOGLE_API_KEY", _api_key)


# --------------------------------------------------------------------------- #
# Cached clients - built once per process instead of once per graph invocation
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=60,
    )


@lru_cache(maxsize=1)
def get_query_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    gemini-embedding-001 emits 3072 dimensions by default. The collection is
    768-dim, so the query vector must be pinned to the same footprint or every
    search is rejected with a dimension-mismatch error.
    """
    kwargs: dict[str, Any] = {
        "model": EMBED_MODEL,
        "task_type": "retrieval_query",
        "google_api_key": resolve_google_api_key(),
    }
    try:
        return GoogleGenerativeAIEmbeddings(output_dimensionality=VECTOR_SIZE, **kwargs)
    except TypeError:
        LOGGER.warning(
            "Installed langchain-google-genai does not expose output_dimensionality; "
            "queries may return %s-dim vectors against a %s-dim collection.",
            "native", VECTOR_SIZE,
        )
        return GoogleGenerativeAIEmbeddings(**kwargs)


@lru_cache(maxsize=4)
def get_chat_model(temperature: float = 0.0) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=temperature,
        google_api_key=resolve_google_api_key(),
    )


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #

class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    current_query: str
    route_to: str
    retrieved_docs: list[dict]
    retrieved_issues: list[dict]
    final_answer: str | None
    needs_clarification: bool
    diagnostics: list[str]          # surfaced to the UI instead of silent failure


class RouterOutput(BaseModel):
    query_type: Literal["conceptual", "api_reference", "troubleshooting"] = Field(
        description="Intent classification"
    )
    needs_clarification: bool = Field(description="True if query is ambiguous")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _search_collection(client: QdrantClient, vector: list[float]) -> list[Any]:
    """query_points() on qdrant-client >= 1.10, search() on older builds."""
    if hasattr(client, "query_points"):
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=TOP_K_DOCS,
            with_payload=True,
        ).points
    return client.search(
        collection_name=COLLECTION_NAME,
        query_vector=vector,
        limit=TOP_K_DOCS,
    )


def _sanitize_github_query(raw: str) -> str:
    """
    Strip characters that GitHub interprets as search qualifiers, collapse
    whitespace, and clamp the length so the q parameter stays valid.
    """
    cleaned = re.sub(r"[\r\n]+", " ", raw)
    cleaned = re.sub(r'[:"()<>\[\]{}]', " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:GITHUB_QUERY_MAX_CHARS].strip()


def _last_user_text(state: AgentState) -> str:
    messages = state.get("messages") or []
    if not messages:
        return state.get("current_query", "")
    content = messages[-1].content
    return content if isinstance(content, str) else str(content)


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #

def router_node(state: AgentState) -> dict:
    query = _last_user_text(state)
    lowered = query.lower()

    if any(kw in lowered for kw in TROUBLESHOOTING_KEYWORDS):
        LOGGER.info("Router: keyword match -> issue_search")
        return {
            "route_to": "issue_search",
            "needs_clarification": False,
            "current_query": query,
        }

    if len(lowered.split()) <= 2:
        LOGGER.info("Router: query too short (%s tokens) -> clarify", len(lowered.split()))
        return {
            "route_to": "clarify",
            "needs_clarification": True,
            "current_query": query,
        }

    try:
        llm = get_chat_model(0.0).with_structured_output(RouterOutput)
        result = llm.invoke(
            [{"role": "system", "content": "Categorize developer intent."}]
            + list(state.get("messages") or [])
        )
        if result.needs_clarification:
            route = "clarify"
        elif result.query_type == "troubleshooting":
            route = "issue_search"
        else:
            route = "doc_qa"
        LOGGER.info("Router: LLM classified as %s -> %s", result.query_type, route)
        return {
            "route_to": route,
            "needs_clarification": result.needs_clarification,
            "current_query": query,
        }
    except Exception as exc:  # network, quota, schema-parse - all recoverable
        LOGGER.warning(
            "Router classification failed (%s: %s); defaulting to doc_qa.",
            type(exc).__name__, exc,
        )
        return {
            "route_to": "doc_qa",
            "needs_clarification": False,
            "current_query": query,
            "diagnostics": [f"Router fallback: {type(exc).__name__}: {exc}"],
        }


def doc_qa_node(state: AgentState) -> dict:
    query = state.get("current_query") or _last_user_text(state)

    try:
        query_vector = get_query_embeddings().embed_query(query)
    except Exception as exc:
        LOGGER.error("Query embedding failed (%s: %s)", type(exc).__name__, exc)
        return {
            "retrieved_docs": [],
            "diagnostics": [f"Embedding error: {type(exc).__name__}: {exc}"],
        }

    if len(query_vector) != VECTOR_SIZE:
        msg = (
            f"Query vector is {len(query_vector)}-dim but collection "
            f"'{COLLECTION_NAME}' expects {VECTOR_SIZE}-dim. "
            "Upgrade langchain-google-genai or rebuild the collection."
        )
        LOGGER.error(msg)
        return {"retrieved_docs": [], "diagnostics": [msg]}

    try:
        client = get_qdrant_client()
        hits = _search_collection(client, query_vector)
    except KeyError as exc:
        LOGGER.error("Missing environment variable: %s", exc)
        return {"retrieved_docs": [], "diagnostics": [f"Missing env var: {exc}"]}
    except (UnexpectedResponse, ApiException) as exc:
        LOGGER.error("Qdrant rejected the search: %s", exc)
        return {"retrieved_docs": [], "diagnostics": [f"Qdrant error: {exc}"]}
    except Exception as exc:
        LOGGER.error("Qdrant search failed (%s: %s)", type(exc).__name__, exc)
        return {
            "retrieved_docs": [],
            "diagnostics": [f"Retrieval error: {type(exc).__name__}: {exc}"],
        }

    docs = []
    for hit in hits:
        payload = hit.payload or {}
        docs.append({
            "text": payload.get("text", ""),
            "source": payload.get("source_url", "unknown"),
            "section": payload.get("section", "General"),
            "score": getattr(hit, "score", None),
        })

    LOGGER.info("doc_qa: %s chunks retrieved for %r", len(docs), query[:60])
    if not docs:
        return {
            "retrieved_docs": [],
            "diagnostics": [
                f"Collection '{COLLECTION_NAME}' returned no matches - is it hydrated?"
            ],
        }
    return {"retrieved_docs": docs}


def issue_search_node(state: AgentState) -> dict:
    query = _sanitize_github_query(state.get("current_query") or _last_user_text(state))
    if not query:
        return {"retrieved_issues": [], "diagnostics": ["Empty GitHub search query."]}

    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    else:
        LOGGER.warning("No GITHUB_TOKEN set - unauthenticated search is limited to 10 req/min.")

    params = {
        "q": f"{query} repo:{GITHUB_REPO} is:issue",
        "sort": "updated",
        "order": "desc",
        "per_page": MAX_ISSUES,
    }

    try:
        response = requests.get(
            GITHUB_SEARCH_URL, headers=headers, params=params, timeout=HTTP_TIMEOUT
        )
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        detail = "rate limited" if status == 403 else "rejected"
        LOGGER.error("GitHub search %s (HTTP %s): %s", detail, status, exc)
        return {
            "retrieved_issues": [],
            "diagnostics": [f"GitHub search HTTP {status} ({detail})."],
        }
    except requests.exceptions.Timeout:
        LOGGER.error("GitHub search timed out after %ss", HTTP_TIMEOUT)
        return {"retrieved_issues": [], "diagnostics": ["GitHub search timed out."]}
    except requests.exceptions.RequestException as exc:
        LOGGER.error("GitHub search transport failure: %s", exc)
        return {"retrieved_issues": [], "diagnostics": [f"GitHub search failed: {exc}"]}
    except ValueError as exc:  # non-JSON body
        LOGGER.error("GitHub search returned malformed JSON: %s", exc)
        return {"retrieved_issues": [], "diagnostics": ["GitHub returned malformed JSON."]}

    issues = [
        {
            "title": item.get("title", "Untitled"),
            "url": item.get("html_url", ""),
            "number": item.get("number"),
            "state": item.get("state", "unknown"),
        }
        for item in payload.get("items", [])[:MAX_ISSUES]
    ]
    LOGGER.info("issue_search: %s issues matched %r", len(issues), query[:60])
    return {"retrieved_issues": issues}


def synthesize_node(state: AgentState) -> dict:
    context_blocks: list[str] = []

    if state.get("retrieved_docs"):
        rendered = "\n".join(
            f"[{d['section']}]({d['source']}): {d['text']}"
            for d in state["retrieved_docs"]
        )
        context_blocks.append("Documentation Context:\n" + rendered)

    if state.get("retrieved_issues"):
        rendered = "\n".join(
            f"Issue #{i['number']} [{i['state']}]: {i['title']} ({i['url']})"
            for i in state["retrieved_issues"]
        )
        context_blocks.append("GitHub Issue Context:\n" + rendered)

    if not context_blocks:
        diagnostics = state.get("diagnostics") or []
        detail = f" (diagnostics: {'; '.join(diagnostics)})" if diagnostics else ""
        LOGGER.info("synthesize: no context available%s", detail)
        return {
            "final_answer": (
                "I could not retrieve any supporting context for that question, so I "
                "would rather not guess. Check that the Qdrant collection is hydrated "
                "and that the GitHub search succeeded." + detail
            )
        }

    system_prompt = (
        "You are an expert developer assistant. Answer the user's prompt using only "
        "the provided context. Strict rule: cite your sources inline using standard "
        "markdown links matching the documentation URLs or issue numbers. If the "
        "context does not cover the question, say so plainly."
    )
    context = "\n\n".join(context_blocks)

    try:
        answer = get_chat_model(0.2).invoke([
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {state.get('current_query', '')}",
            },
        ]).content
    except Exception as exc:
        LOGGER.error("Synthesis failed (%s: %s)", type(exc).__name__, exc)
        return {
            "final_answer": f"Synthesis failed: {type(exc).__name__}: {exc}",
            "diagnostics": (state.get("diagnostics") or []) + [str(exc)],
        }

    return {"final_answer": answer}


def clarify_node(state: AgentState) -> dict:
    return {
        "final_answer": (
            "Could you please provide more context or add the specific error "
            "signature you are running into?"
        )
    }


def route_decision(state: AgentState) -> str:
    return state.get("route_to", "doc_qa")


# --------------------------------------------------------------------------- #
# Graph assembly
# --------------------------------------------------------------------------- #

graph = StateGraph(AgentState)
graph.add_node("router", router_node)
graph.add_node("doc_qa", doc_qa_node)
graph.add_node("issue_search", issue_search_node)
graph.add_node("synthesize", synthesize_node)
graph.add_node("clarify", clarify_node)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    route_decision,
    {"doc_qa": "doc_qa", "issue_search": "issue_search", "clarify": "clarify"},
)
graph.add_edge("doc_qa", "synthesize")
graph.add_edge("issue_search", "synthesize")
graph.add_edge("synthesize", END)
graph.add_edge("clarify", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    # Smoke test: python engine.py
    import uuid as _uuid

    from langchain_core.messages import HumanMessage

    probe = "How do I add a checkpointer to a LangGraph StateGraph?"
    cfg = {"configurable": {"thread_id": str(_uuid.uuid4())}}
    result = app.invoke({"messages": [HumanMessage(content=probe)]}, cfg)
    print("ROUTE:", result.get("route_to"))
    print("DOCS:", len(result.get("retrieved_docs") or []))
    print("ISSUES:", len(result.get("retrieved_issues") or []))
    print("DIAGNOSTICS:", result.get("diagnostics"))
    print("ANSWER:\n", result.get("final_answer"))

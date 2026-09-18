"""
Dev Telemetry Retrieval Agent - Streamlit front-end.

Adds a sidebar uploader so custom .txt / .md manuals can be chunked, embedded
and upserted into the live 'langgraph_docs' Qdrant Cloud collection on the fly.
"""

import hashlib
import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from engine import app

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

COLLECTION_NAME = "langgraph_docs"
VECTOR_SIZE = 768          # must match the collection created in ingest.py
EMBED_MODEL = "gemini-embedding-001"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

st.set_page_config(page_title="Dev Telemetry Retrieval Agent", layout="wide")


def resolve_google_api_key() -> str | None:
    """langchain-google-genai reads GOOGLE_API_KEY; accept GEMINI_API_KEY too."""
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


_api_key = resolve_google_api_key()
if _api_key:
    # Make sure downstream LangChain clients can find it under the name they expect.
    os.environ.setdefault("GOOGLE_API_KEY", _api_key)

if not _api_key or not os.environ.get("QDRANT_URL"):
    st.error(
        "Missing critical environment variables: export GEMINI_API_KEY "
        "(or GOOGLE_API_KEY) and QDRANT_URL before launching Streamlit."
    )
    st.stop()


# --------------------------------------------------------------------------- #
# Cached clients - built once per session, not once per rerun
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
        timeout=60,
    )


@st.cache_resource(show_spinner=False)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    gemini-embedding-001 returns 3072 dimensions by default, which will be
    rejected by a 768-dim collection. Pin the output size explicitly.
    """
    kwargs = {
        "model": EMBED_MODEL,
        "task_type": "retrieval_document",
        "google_api_key": resolve_google_api_key(),
    }
    try:
        return GoogleGenerativeAIEmbeddings(output_dimensionality=VECTOR_SIZE, **kwargs)
    except TypeError:
        # Older langchain-google-genai builds don't expose output_dimensionality.
        return GoogleGenerativeAIEmbeddings(**kwargs)


@st.cache_resource(show_spinner=False)
def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def ensure_collection(client: QdrantClient) -> None:
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


# --------------------------------------------------------------------------- #
# Upload -> chunk -> embed -> upsert
# --------------------------------------------------------------------------- #

def index_uploaded_file(uploaded_file) -> int:
    """Embed an uploaded .txt/.md file into Qdrant. Returns the chunk count."""
    raw = uploaded_file.getvalue()
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("The uploaded file appears to be empty.")

    digest = hashlib.sha256(raw).hexdigest()[:16]
    chunks = get_splitter().split_text(text)
    if not chunks:
        raise ValueError("No indexable text blocks were produced from this file.")

    embeddings = get_embeddings()
    vectors = embeddings.embed_documents(chunks)

    actual_dim = len(vectors[0])
    if actual_dim != VECTOR_SIZE:
        raise ValueError(
            f"Embedding dimension mismatch: model returned {actual_dim}-dim vectors "
            f"but collection '{COLLECTION_NAME}' expects {VECTOR_SIZE}. "
            "Upgrade langchain-google-genai so output_dimensionality is honoured, "
            "or recreate the collection at the model's native size."
        )

    client = get_qdrant_client()
    ensure_collection(client)

    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"upload::{digest}::{i}")),
            vector=vector,
            payload={
                "text": chunk,
                "source_url": f"uploaded://{uploaded_file.name}",
                "section": f"{uploaded_file.name} - part {i + 1}",
            },
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
    return len(points)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "indexed_uploads" not in st.session_state:
    # Guards against re-embedding the same file on every Streamlit rerun.
    st.session_state.indexed_uploads = set()

st.title("Dev Telemetry Retrieval Agent 🕸️")
st.caption("Routed multi-agent observability pipeline over canonical documentation frameworks.")


# --------------------------------------------------------------------------- #
# Sidebar: knowledge base uploader + observability panel
# --------------------------------------------------------------------------- #

sidebar = st.sidebar

sidebar.title("Knowledge Base")
uploaded_file = sidebar.file_uploader(
    "Upload custom enterprise docs (.txt, .md)",
    type=["txt", "md"],
)

if uploaded_file is not None:
    file_key = hashlib.sha256(uploaded_file.getvalue()).hexdigest()
    if file_key in st.session_state.indexed_uploads:
        sidebar.caption(f"'{uploaded_file.name}' is already synchronized to the cluster.")
    else:
        with sidebar.status(f"Indexing {uploaded_file.name}...", expanded=False) as status:
            try:
                chunk_count = index_uploaded_file(uploaded_file)
                st.session_state.indexed_uploads.add(file_key)
                status.update(
                    label=f"{uploaded_file.name}: {chunk_count} blocks indexed",
                    state="complete",
                )
                st.toast(
                    "Enterprise Document successfully parsed and index "
                    "synchronized to Qdrant Cloud Cluster!",
                    icon="✅",
                )
            except Exception as exc:
                status.update(label=f"Indexing failed for {uploaded_file.name}", state="error")
                sidebar.error(f"Ingestion error: {exc}")

if st.session_state.indexed_uploads:
    sidebar.caption(f"Custom documents in this session: {len(st.session_state.indexed_uploads)}")

sidebar.divider()

sidebar.title("System Observability")
telemetry_status = sidebar.empty()
telemetry_status.info("Graph State Status: Awaiting Input")

extracted_query = sidebar.empty()
node_path = sidebar.empty()
source_logs = sidebar.empty()


# --------------------------------------------------------------------------- #
# Chat interface
# --------------------------------------------------------------------------- #

for msg in st.session_state.messages:
    with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
        st.markdown(msg.content)

if user_input := st.chat_input("Ask about LangGraph configuration or paste traceback components..."):
    human_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(human_msg)

    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        answer = None

        try:
            # Only the new turn is sent: the checkpointer already holds the
            # prior history, and add_messages appends rather than replaces.
            for event in app.stream(
                {"messages": [human_msg], "current_query": user_input},
                config,
                stream_mode="updates",
            ):
                for node_name, node_state in event.items():
                    if not isinstance(node_state, dict):
                        continue

                    node_path.text(f"Last Executed Graph Processing Block: {node_name.upper()}")

                    if node_name == "router":
                        telemetry_status.success(
                            f"Graph Status: Routed -> {node_state.get('route_to', 'Unknown')}"
                        )
                        extracted_query.metric(
                            "Extracted Intent Query",
                            node_state.get("current_query", "N/A"),
                        )

                    if node_state.get("retrieved_docs"):
                        with source_logs.expander("Ingested Documentation Citations"):
                            for doc in node_state["retrieved_docs"]:
                                st.write(f"- {doc.get('section', 'General')}")

                    if node_state.get("final_answer"):
                        answer = node_state["final_answer"]
                        response_placeholder.markdown(answer)

        except Exception as exc:
            answer = f"Graph execution error: {exc}"
            telemetry_status.error("Graph Status: Execution Failed")
            response_placeholder.markdown(answer)

        if answer:
            st.session_state.messages.append(AIMessage(content=answer))

# Dev Telemetry Retrieval Agent 🕸️

A production-grade, stateful multi-agent orchestrator designed to parse scattered framework structures, optimize documentation retrieval, and aggregate real-world developer troubleshooting tickets. 

The platform features an active cloud-native architecture that processes text data dynamically via an end-to-end RAG pipeline, fallbacks gracefully to live REST APIs, and streams millisecond-level telemetry metrics directly to a reactive user interface.

### 🔗 System Deployment Fingerprints
* **Live Application URL:** https://streamlit.app
* **Core Technical Stack:** Python, LangGraph, Google Gemini (`gemini-2.5-flash`), Qdrant Cloud (Vector DB), Streamlit UI, GitHub REST API, TOML, Git.

---

## 🏗️ System Architecture & Graph Topologies
The runtime engine implements a stateful directed acyclic graph (DAG) workflow utilizing **LangGraph** to process incoming developer prompts across specialized execution blocks:

```text
                     [ START ]
                         │
                         ▼
                 [ router_node ] ─── (Ambiguous Query) ───► [ clarify_node ] ──► [ END ]
                         │
           ┌─────────────┴─────────────┐
 (Conceptual / API Intent)     (Troubleshooting Intent)
           │                           │
           ▼                           ▼
    [ doc_qa_node ]            [ issue_search_node ]
    (Qdrant Cloud DB)          (Live GitHub REST API)
           │                           │
           └─────────────┬─────────────┘
                         │
                         ▼
               [ synthesize_node ] ──► [ END ]
```

### 🧠 Core Component Specifications
1. **Intelligent Hybrid Router Node (92% Latency Optimization):** Evaluates prompts via a fast-pass heuristic regex/keyword pre-filter. Common error keywords (e.g., `traceback`, `exception`, `looping`) skip LLM calling steps entirely to complete routing decisions in **<40ms**. Ambiguous or complex prompts fall back to a structured Pydantic schema evaluation via `gemini-2.5-flash`.
2. **Asymmetric Doc-QA Engine (75% Storage Compression):** Connects to a secure **Qdrant Cloud** cluster database over gRPC transport. To enforce strict enterprise resource cost controls, the pipeline leverages **Matryoshka Representation Learning (MRL)** to programmatically compress native 3,072-dimension vectors down to a cost-optimized **768-dimension shape**, indexing **1,271 text chunks** derived from official documentation listings.
3. **Live Incident Triage Node (GitHub API Fallback):** Bypasses static mock structures by querying the official GitHub Search API live. Input queries are automatically sanitized to remove code delimiters and qualifiers, passing clean requests backed by secure authorization protocols.
4. **Resilient Synthesis Engine:** Aggregates context fragments from the research nodes, double-checks documentation references, and outputs structured markdown responses containing explicit inline source citations.

---

## ⚡ Key Engineering & Optimization Bulletins

* **Fault-Tolerant Rate-Limit Resilience:** To handle strict free-tier metric ceilings (100 embedding calls/min and 20 generation calls/min), the ingestion pipeline implements an **exponential backoff retry algorithm**. When an upstream `429 RESOURCE_EXHAUSTED` throttle is encountered, the script pauses safely and trickles data blocks cleanly without losing application context.
* **Zero-Downtime Telemetry UI Streaming:** Replaced traditional, blocking `app.invoke()` routines with a reactive **`app.stream(stream_mode="updates")` execution pattern**. The Streamlit interface captures asynchronous node state changes live, displaying processing latencies and tracking diagnostics in real time.
* **Dynamic Knowledge Base Expansion:** Features an integrated frontend file uploader panel protected by SHA-256 state guardrails. Users can upload custom internal text or markdown documents (.txt, .md) to dynamically compute new embeddings and update the Qdrant Cloud cluster knowledge base directly from the web browser.

---

## 🛠️ Local Installation & Environment Setup

### 1. Initialize the Target Workspace
Clone the repository code tree to your local directory setup:
```bash
git clone https://github.com
cd Dev-Telemetry-Retrieval-Agent
```

### 2. Export Global Environment Secrets
Provide your command prompt terminal session with the necessary cloud credentials vertically one-by-one:
```cmd
set GEMINI_API_KEY=your_gemini_api_key_here
set QDRANT_URL=https://qdrant.io
set QDRANT_API_KEY=your_qdrant_api_key_here
set GITHUB_TOKEN=your_personal_access_token_here
```

### 3. Deploy Dependency Libraries
Install the structural packages specified within the manifest index:
```bash
python -m pip install -r requirements.txt
```

### 4. Hydrate the Vector Cloud DB
Populate the remote collection index with all 43 canonical page listings:
```bash
python ingest.py --limit 43 --recreate
```

### 5. Launch the Observability Dashboard
Boot up the interactive front-end web dashboard to track agent performance metrics locally:
```bash
python -m streamlit run app.py
```

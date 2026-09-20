"""
Phase 7 -- Deployment & Demo.

Streamlit UI for the RAG pipeline. Run locally with:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud (streamlit.io/cloud) or a
Hugging Face Space: just point it at this repo and set GEMINI_API_KEY as a
secret -- no server, Docker, or paid hosting required.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

import config
from src.pipeline import RAGPipeline

st.set_page_config(page_title="Domain RAG — Meridian Systems Handbook", page_icon="🔎", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17272b;
        --muted: #68777a;
        --paper: #f5f7f5;
        --panel: #ffffff;
        --line: #dfe7e3;
        --teal: #123b3d;
        --teal-soft: #dcebea;
        --amber: #f2b84b;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'Manrope', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stDecoration"] { display: none; }
    .block-container { max-width: 1180px; padding: 3rem 4rem 5rem; }

    [data-testid="stSidebar"] > div:first-child {
        background: var(--teal);
        border-right: 1px solid #295657;
        padding: 2rem 1.25rem;
    }
    [data-testid="stSidebar"] * { color: #edf5f1; }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #a9c2be; }
    [data-testid="stSidebar"] hr { border-color: #2d5b5b; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: #1d4c4d;
        border: 1px solid #37696a;
        color: #f7fbf9;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
        border-color: var(--amber);
        background: #285d5d;
        color: #fffdf7;
    }
    [data-testid="stSidebar"] [data-testid="stJson"] {
        background: #0e3032;
        border: 1px solid #2c5b5b;
        border-radius: 6px;
    }
    [data-testid="stSidebar"] [data-testid="stJson"] * {
        color: #173b3d !important;
        fill: #173b3d !important;
    }
    [data-testid="stSidebar"] [data-testid="stJson"] {
        background: #ffffff !important;
        color: #173b3d !important;
        padding: 0.65rem;
    }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
            color: #edf5f1 !important;
        }
        [data-testid="stSidebar"] [data-testid="stAlert"] {
            background: #e6f0ef !important;
            border: 1px solid #a9c4bf !important;
        }
        [data-testid="stSidebar"] [data-testid="stAlert"] p {
            color: #173b3d !important;
        }

    [data-testid="stButton"] button {
        background: var(--teal) !important;
        border: 1px solid var(--teal) !important;
        color: #ffffff !important;
        min-height: 2.6rem;
    }
    [data-testid="stButton"] button p,
    [data-testid="stButton"] button span,
    [data-testid="stButton"] button div { color: #ffffff !important; }
    [data-testid="stButton"] button:hover {
        background: #285d5d !important;
        border-color: #285d5d !important;
        color: #ffffff !important;
    }
    [data-testid="stButton"] button:focus-visible {
        box-shadow: 0 0 0 3px rgba(242, 184, 75, 0.55) !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: #1d4c4d !important;
        border-color: #5b8582 !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button p,
    [data-testid="stSidebar"] [data-testid="stButton"] button span {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button div { color: #ffffff !important; }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p { color: var(--ink) !important; }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary p { color: #ffffff !important; }
    [data-baseweb="select"] > div,
    [data-baseweb="select"] input { color: var(--ink) !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div { color: var(--ink) !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] input {
            -webkit-text-fill-color: var(--ink) !important;
        }
    [data-testid="stSidebar"] [data-baseweb="select"] > div { background: #ffffff !important; }
    [data-baseweb="select"] svg { fill: #587073 !important; }
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="menu"] [role="option"] {
        background: #ffffff !important;
        color: var(--ink) !important;
    }
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="menu"] [role="option"]:hover { background: var(--teal-soft) !important; }

    h1, h2, h3, p, label { font-family: 'Manrope', sans-serif; }
    h1 { color: var(--ink); font-size: 2.5rem; letter-spacing: -0.02em; line-height: 1.1; }
    h2, h3 { color: var(--ink); }
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stText"] { color: var(--ink) !important; }
    [data-testid="stAlert"] p,
    [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

    .eyebrow {
        color: #4e7f7c;
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
    }
    .hero-copy { max-width: 720px; margin-bottom: 2rem; }
    .hero-copy h1 { margin: 0 0 0.75rem; }
    .hero-copy p { color: var(--muted); font-size: 1rem; margin: 0; }

    [data-testid="stChatMessage"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        margin: 0.8rem 0;
        padding: 1rem 1.15rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { line-height: 1.7; }
    [data-testid="stChatMessage"] [data-testid="stExpander"] {
        background: #f8faf8;
        border: 1px solid var(--line);
        border-radius: 6px;
        margin-top: 1rem;
    }
    [data-testid="stChatInput"] { border-color: #a9c4bf; background: #ffffff; }
    [data-testid="stChatInput"] textarea {
        color: var(--ink) !important;
        -webkit-text-fill-color: var(--ink) !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #68777a !important;
        opacity: 1 !important;
    }
    [data-testid="stChatInput"] button svg { fill: var(--teal) !important; }
    [data-testid="stChatInput"]:focus-within { border-color: var(--teal); box-shadow: 0 0 0 1px var(--teal); }
    [data-testid="stText"] { font-size: 0.82rem; line-height: 1.55; }
    .source-meta { color: var(--muted); font-size: 0.78rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_pipeline(api_key: str, embedder_backend: str) -> RAGPipeline:
    pipeline_key = (api_key, embedder_backend)
    if "pipeline" not in st.session_state or st.session_state.get("pipeline_key") != pipeline_key:
        pipeline = RAGPipeline(api_key=api_key, embedder_backend=embedder_backend)
        if pipeline.index_exists() and pipeline.index_backend() == embedder_backend:
            pipeline.load()
        st.session_state.pipeline = pipeline
        st.session_state.pipeline_key = pipeline_key
    return st.session_state.pipeline


def get_deployment_api_key() -> str:
    """Read the hidden deployment secret, with .env as a local fallback."""
    try:
        return st.secrets.get("GEMINI_API_KEY", "") or config.GEMINI_API_KEY
    except Exception:
        return config.GEMINI_API_KEY


def main():
    st.markdown(
        """
        <div class="hero-copy">
            <div class="eyebrow">Meridian Systems / Knowledge Workspace</div>
            <h1>Engineering handbook, made searchable.</h1>
            <p>Ask a question and receive a concise answer grounded in the indexed handbook, with the evidence shown alongside it.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("<div class='eyebrow'>Workspace controls</div>", unsafe_allow_html=True)
        st.header("Configuration")
        api_key = get_deployment_api_key()
        if not api_key:
            st.error("Gemini API key is not configured for this deployment.")
            st.stop()
        st.success("Gemini API is configured securely.")

        embedder_backend = st.selectbox(
            "Embedding backend",
            ["local", "gemini"],
            index=0,
            help="Local embeddings avoid Gemini embedding quotas. Gemini is useful for benchmarking but has a free-tier request limit.",
        )
        pipeline = get_pipeline(api_key, embedder_backend)

        st.divider()
        st.subheader("Knowledge index")
        chunk_strategy = st.selectbox("Chunking strategy", ["recursive", "fixed_size", "semantic"], index=0)
        if st.button("Build / rebuild index from data/sample_docs", use_container_width=True):
            try:
                with st.spinner("Ingesting documents, embedding, and indexing..."):
                    num_chunks = pipeline.ingest(chunk_strategy=chunk_strategy)
                    pipeline.save()
                st.success(f"Indexed {num_chunks} chunks using {embedder_backend} embeddings.")
            except Exception as exc:
                st.error(f"Index build failed: {exc}")

        if pipeline.hybrid is None:
            st.info("No index loaded yet. Click the button above to build one.")

        st.divider()
        st.subheader("Session telemetry")
        st.json(pipeline.cost_tracker.summary())

    if pipeline.hybrid is None:
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("warnings"):
                for w in turn["warnings"]:
                    st.warning(w)
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for i, sc in enumerate(turn["sources"], start=1):
                    st.markdown(f"**[{i}] {sc.chunk.source}** "
                                f"{'· ' + sc.chunk.section_header if sc.chunk.section_header else ''}")
                    st.text(sc.chunk.text[:500])

    question = st.chat_input("Ask a question about the handbook...")
    if question:
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and generating..."):
                response = pipeline.query(question)
            st.write(response["answer"])
            if response.get("cache_hit"):
                st.caption("⚡ served from cache")
            for w in response.get("warnings", []):
                st.warning(w)
            with st.expander(f"Sources ({len(response['retrieved_chunks'])})"):
                for i, sc in enumerate(response["retrieved_chunks"], start=1):
                    score = f"{sc.rerank_score:.2f}" if sc.rerank_score is not None else "n/a"
                    st.markdown(f"**[{i}] {sc.chunk.source}** "
                                f"{'· ' + sc.chunk.section_header if sc.chunk.section_header else ''} "
                                f"· rerank score: {score}")
                    st.text(sc.chunk.text[:500])

        st.session_state.history.append({
            "question": question,
            "answer": response["answer"],
            "sources": response["retrieved_chunks"],
            "warnings": response.get("warnings", []),
        })


if __name__ == "__main__":
    main()

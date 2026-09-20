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


def get_pipeline(api_key: str, embedder_backend: str) -> RAGPipeline:
    pipeline_key = (api_key, embedder_backend)
    if "pipeline" not in st.session_state or st.session_state.get("pipeline_key") != pipeline_key:
        pipeline = RAGPipeline(api_key=api_key, embedder_backend=embedder_backend)
        if pipeline.index_exists() and pipeline.index_backend() == embedder_backend:
            pipeline.load()
        st.session_state.pipeline = pipeline
        st.session_state.pipeline_key = pipeline_key
    return st.session_state.pipeline


def main():
    st.title("🔎 Domain-Specific RAG — Engineering Handbook Q&A")
    st.caption(
        "Ask questions about Meridian Systems' (fictional) internal engineering handbook. "
        "Answers are grounded only in the retrieved documents, with citations."
    )

    with st.sidebar:
        st.header("Setup")
        api_key = st.text_input(
            "Gemini API key",
            value=config.GEMINI_API_KEY,
            type="password",
            help="Free key: https://aistudio.google.com/apikey",
        )
        if not api_key:
            st.warning("Enter a Gemini API key to continue.")
            st.stop()

        embedder_backend = st.selectbox(
            "Embedding backend",
            ["local", "gemini"],
            index=0,
            help="Local embeddings avoid Gemini embedding quotas. Gemini is useful for benchmarking but has a free-tier request limit.",
        )
        pipeline = get_pipeline(api_key, embedder_backend)

        st.divider()
        st.subheader("Index")
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
        st.subheader("Session stats")
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

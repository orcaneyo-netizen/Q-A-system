# src/app.py

"""Simple Streamlit UI for the Q&A system.

Provides a text input field where the user can type a query, hits a button
to perform retrieval using the HybridRetriever, and displays the top‑k
retrieved document snippets.
"""

import streamlit as st
from src.retrieval.retriever import retriever  # The singleton HybridRetriever

st.set_page_config(page_title="Q&A Retrieval Demo", layout="centered")

st.title("🧠 Q&A Retrieval Demo")
st.caption("Enter a question and see the top retrieved passages.")

query = st.text_input("Your query", placeholder="Ask something...")
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.number_input("Top‑K", min_value=1, max_value=20, value=5, step=1)
    use_hybrid = st.checkbox("Use hybrid (dense + BM25)", value=True)
with col2:
    pass

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving documents…"):
            docs = retriever.retrieve(query, top_k=top_k, use_hybrid=use_hybrid)
        if not docs:
            st.info("No documents found.")
        else:
            st.success(f"Found {len(docs)} document(s).")
            for i, doc in enumerate(docs, start=1):
                st.markdown(f"**Result {i}:**")
                st.code(doc.page_content, language="markdown")
                if doc.metadata:
                    st.json(doc.metadata)
                st.divider()

st.sidebar.header("About")
st.sidebar.info(
    "This demo uses a \"HybridRetriever\" that combines dense Chroma similarity "
    "with sparse BM25 retrieval. Adjust the settings in the sidebar to experiment."
)

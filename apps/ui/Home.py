import os

import requests
import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Nutritional RAG", layout="wide")

    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8001").rstrip("/")

    st.title("Nutritional RAG")
    st.caption("Ask nutrition questions and inspect grounded citations from your vector index.")

    with st.form("query-form"):
        question = st.text_area(
            "Question",
            value="What is the best way to structure a healthy meal plan for a resistance training athlete according to the World Health Organization?",
            height=120,
        )
        top_k = st.slider("Top K retrieved chunks", min_value=1, max_value=10, value=5)
        rerank_candidate_multiplier = st.slider(
            "Rerank candidate multiplier",
            min_value=1,
            max_value=20,
            value=10,
            help="Higher values pull more Pinecone candidates before reranking."
            " This improves source diversity but increases latency.",
        )
        submitted = st.form_submit_button("Ask")

    if not submitted:
        st.info("Submit a question to query the API and view citations.")
        return

    if not question.strip():
        st.warning("Please enter a question.")
        return

    try:
        response = requests.post(
            f"{api_base_url}/query",
            json={
                "question": question.strip(),
                "top_k": top_k,
                "rerank_candidate_multiplier": rerank_candidate_multiplier,
            },
            timeout=120,
        )
    except requests.RequestException as error:
        st.error(f"Request failed: {error}")
        return

    if response.status_code != 200:
        st.error(f"API error {response.status_code}: {response.text}")
        return

    payload = response.json()
    st.subheader("Answer")
    st.write(payload.get("answer", "No answer generated."))

    cache_hit = bool(payload.get("cache_hit", False))
    st.caption(f"Cache hit: {'yes' if cache_hit else 'no'}")

    citations = payload.get("citations", [])
    st.subheader("Sources")
    if not citations:
        st.write("No citations returned.")
        return

    for index, citation in enumerate(citations, start=1):
        source = citation.get("source_id") or "unknown-source"
        doc = citation.get("document_id") or "unknown-doc"
        page = citation.get("page_number")
        chunk_index = citation.get("chunk_index")
        score = citation.get("score")
        title = citation.get("title")
        pubmed_url = citation.get("pubmed_url")
        youtube_url = citation.get("youtube_url")
        harvard_url = citation.get("harvard_url")

        page_text = f"page {page}" if page is not None else "page n/a"
        chunk_text = f"chunk {chunk_index}" if chunk_index is not None else "chunk n/a"
        title_text = f"{title}" if title else source
        source_badge = ""
        if pubmed_url:
            source_badge = f" | [PubMed]({pubmed_url})"
        if youtube_url:
            source_badge += f" | [YouTube]({youtube_url})"
        if harvard_url:
            source_badge += f" | [Harvard]({harvard_url})"

        citation_line = (
            f"{index}. **{title_text}** | {page_text} | {chunk_text} "
            f"| score={score:.4f}{source_badge}"
        )
        st.markdown(citation_line)
        st.caption(doc)


if __name__ == "__main__":
    main()

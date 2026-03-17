import os

import streamlit as st


def main() -> None:
    st.set_page_config(page_title="Nutritional RAG", layout="wide")

    api_base_url = os.getenv("API_BASE_URL", "http://localhost:8000")

    st.title("Nutritional RAG")
    st.caption("Repository scaffold for the retrieval, reranking, serving, and demo stack.")

    left, right = st.columns(2)

    with left:
        st.subheader("Current Scope")
        st.write(
            "This UI is a placeholder while the retrieval pipeline, reranker, "
            "and serving layer are being wired in."
        )
        st.code(f"API base URL: {api_base_url}")

    with right:
        st.subheader("Planned Components")
        st.markdown(
            "\n".join(
                [
                    "- Pinecone-backed retrieval",
                    "- GPT-4 answer generation",
                    "- PyTorch reranking",
                    "- Redis cache metrics",
                    "- MLflow experiment tracking",
                ]
            )
        )


if __name__ == "__main__":
    main()

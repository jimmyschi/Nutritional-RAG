import sys
import types

from nutritional_rag.etl.extract import extract_source
from nutritional_rag.etl.models import ExtractionSource


def test_extract_html_source() -> None:
    source = ExtractionSource(
        source_id="article",
        kind="html",
        location="tests/fixtures/extract/article.html",
        source_name="article",
    )

    docs = extract_source(source)

    assert len(docs) == 1
    assert "Hydration Basics" in docs[0].text
    assert docs[0].source_id == "article"


def test_extract_pubmed_source_with_loader_mock(monkeypatch) -> None:
    fake_child = types.ModuleType("langchain_community.document_loaders")
    fake_parent = types.ModuleType("langchain_community")

    class _FakeDocument:
        def __init__(self, page_content: str, metadata: dict[str, str]) -> None:
            self.page_content = page_content
            self.metadata = metadata

    class _FakePubMedLoader:
        def __init__(self, query: str, load_max_docs: int) -> None:
            self.query = query
            self.load_max_docs = load_max_docs

        def load(self):
            assert self.query == "creatine supplementation performance"
            assert self.load_max_docs == 2
            return [
                _FakeDocument(
                    page_content="Creatine may improve short-duration high-intensity performance.",
                    metadata={"Title": "Creatine and Performance", "uid": "12345"},
                ),
                _FakeDocument(
                    page_content="",
                    metadata={"Title": "Empty Result"},
                ),
            ]

    fake_child.PubMedLoader = _FakePubMedLoader
    fake_parent.document_loaders = fake_child
    monkeypatch.setitem(sys.modules, "langchain_community", fake_parent)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", fake_child)

    source = ExtractionSource(
        source_id="pubmed-nutrition",
        kind="pubmed",
        location="creatine supplementation performance",
        source_name="PubMed",
        metadata={"load_max_docs": 2, "domain": "sports-nutrition"},
    )

    docs = extract_source(source)

    assert len(docs) == 1
    assert docs[0].title == "Creatine and Performance"
    assert docs[0].source_id == "pubmed-nutrition"
    assert docs[0].metadata["domain"] == "sports-nutrition"
    assert docs[0].metadata["uid"] == "12345"
    assert docs[0].metadata["query"] == "creatine supplementation performance"

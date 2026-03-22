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


def test_extract_pubmed_title_dict_metadata_is_normalized(monkeypatch) -> None:
    fake_child = types.ModuleType("langchain_community.document_loaders")
    fake_parent = types.ModuleType("langchain_community")

    class _FakeDocument:
        def __init__(self, page_content: str, metadata: dict[str, object]) -> None:
            self.page_content = page_content
            self.metadata = metadata

    class _FakePubMedLoader:
        def __init__(self, query: str, load_max_docs: int) -> None:
            self.query = query
            self.load_max_docs = load_max_docs

        def load(self):
            return [
                _FakeDocument(
                    page_content="Review paper text.",
                    metadata={
                        "Title": {
                            "i": "Cordyceps militaris",
                            "#text": "Current Evidence in Humans",
                        },
                        "uid": "abc",
                    },
                )
            ]

    fake_child.PubMedLoader = _FakePubMedLoader
    fake_parent.document_loaders = fake_child
    monkeypatch.setitem(sys.modules, "langchain_community", fake_parent)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", fake_child)

    source = ExtractionSource(
        source_id="pubmed-nutrition",
        kind="pubmed",
        location="ergogenic aids",
        source_name="PubMed",
        metadata={"load_max_docs": 1},
    )

    docs = extract_source(source)

    assert len(docs) == 1
    assert docs[0].title == "Cordyceps militaris Current Evidence in Humans"


def test_extract_youtube_source_with_loader_mock(monkeypatch) -> None:
    fake_child = types.ModuleType("langchain_community.document_loaders")
    fake_parent = types.ModuleType("langchain_community")

    class _FakeDocument:
        def __init__(self, page_content: str, metadata: dict[str, object]) -> None:
            self.page_content = page_content
            self.metadata = metadata

    class _FakeYoutubeLoader:
        def __init__(self, docs):
            self._docs = docs

        def load(self):
            return self._docs

        @classmethod
        def from_youtube_url(cls, url: str, add_video_info: bool, language: list[str]):
            assert url == "https://www.youtube.com/watch?v=LrHcCdKgdiE"
            assert add_video_info is False
            assert language == ["en"]
            return cls(
                [
                    _FakeDocument(
                        page_content=(
                            "Micronutrients support metabolic health and insulin response."
                        ),
                        metadata={
                            "title": "Nutrition Basics for Metabolic Health",
                            "source": url,
                            "view_count": 12345,
                        },
                    )
                ]
            )

    fake_child.YoutubeLoader = _FakeYoutubeLoader
    fake_parent.document_loaders = fake_child
    monkeypatch.setitem(sys.modules, "langchain_community", fake_parent)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", fake_child)

    source = ExtractionSource(
        source_id="youtube-nutrition-video",
        kind="youtube",
        location="https://www.youtube.com/watch?v=LrHcCdKgdiE",
        source_name="YouTube",
        metadata={"domain": "general-nutrition", "add_video_info": False, "language": ["en"]},
    )

    docs = extract_source(source)

    assert len(docs) == 1
    assert docs[0].source_id == "youtube-nutrition-video"
    assert docs[0].title == "Nutrition Basics for Metabolic Health"
    assert docs[0].metadata["domain"] == "general-nutrition"
    assert docs[0].metadata["video_url"] == "https://www.youtube.com/watch?v=LrHcCdKgdiE"


def test_extract_web_source_with_loader_mock(monkeypatch) -> None:
    fake_child = types.ModuleType("langchain_community.document_loaders")
    fake_parent = types.ModuleType("langchain_community")

    class _FakeDocument:
        def __init__(self, page_content: str, metadata: dict[str, object]) -> None:
            self.page_content = page_content
            self.metadata = metadata

    class _FakeWebBaseLoader:
        def __init__(self, web_paths: list[str], header_template: dict[str, str] | None = None) -> None:
            assert web_paths == ["https://nutritionsource.hsph.harvard.edu/nutrition-news/"]
            assert header_template is not None
            assert "User-Agent" in header_template
            self._web_paths = web_paths

        def load(self):
            return [
                _FakeDocument(
                    page_content=(
                        "Nutrition News: Recent findings on healthy dietary patterns."
                    ),
                    metadata={
                        "title": "Nutrition News",
                        "source": self._web_paths[0],
                    },
                )
            ]

    fake_child.WebBaseLoader = _FakeWebBaseLoader
    fake_parent.document_loaders = fake_child
    monkeypatch.setitem(sys.modules, "langchain_community", fake_parent)
    monkeypatch.setitem(sys.modules, "langchain_community.document_loaders", fake_child)

    source = ExtractionSource(
        source_id="harvard-nutrition-news",
        kind="web",
        location="https://nutritionsource.hsph.harvard.edu/nutrition-news/",
        source_name="Harvard Nutrition Source",
        metadata={"domain": "general-nutrition", "source_type": "web-news"},
    )

    docs = extract_source(source)

    assert len(docs) == 1
    assert docs[0].source_id == "harvard-nutrition-news"
    assert docs[0].title == "Nutrition News"
    assert docs[0].metadata["domain"] == "general-nutrition"
    assert docs[0].metadata["url"] == "https://nutritionsource.hsph.harvard.edu/nutrition-news/"

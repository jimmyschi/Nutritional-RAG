from nutritional_rag.etl.extract import extract_source
from nutritional_rag.etl.models import ExtractionSource


def test_extract_html_source() -> None:
    source = ExtractionSource(
        source_id="article",
        kind="html",
        location="etl/samples/article.html",
        source_name="article",
    )

    docs = extract_source(source)

    assert len(docs) == 1
    assert "Hydration Basics" in docs[0].text
    assert docs[0].source_id == "article"

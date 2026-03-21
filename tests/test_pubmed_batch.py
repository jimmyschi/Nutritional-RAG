from nutritional_rag.etl.models import RawDocument
from nutritional_rag.etl.pubmed_batch import (
    PubMedBatchConfig,
    _build_pubmed_source,
    _document_dedupe_key,
)


def test_pubmed_batch_config_accepts_string_topics() -> None:
    config = PubMedBatchConfig.model_validate(
        {
            "topics": [
                "vitamin d insulin resistance adults",
                {"query": "magnesium blood pressure adults", "topic_id": "bp-magnesium"},
            ]
        }
    )

    assert len(config.topics) == 2
    assert config.topics[0].query == "vitamin d insulin resistance adults"
    assert config.topics[1].topic_id == "bp-magnesium"


def test_build_pubmed_source_generates_source_id_and_metadata() -> None:
    config = PubMedBatchConfig.model_validate(
        {
            "topics": ["vitamin d insulin resistance adults"],
            "load_max_docs": 7,
        }
    )

    source = _build_pubmed_source(config, config.topics[0])

    assert source.kind == "pubmed"
    assert source.source_id.startswith("pubmed-topic-vitamin-d-insulin-resistance")
    assert source.metadata["load_max_docs"] == 7
    assert source.metadata["topic_query"] == "vitamin d insulin resistance adults"


def test_document_dedupe_key_prefers_pubmed_uid() -> None:
    document = RawDocument(
        document_id="doc-1",
        source_id="pubmed-topic-vitamin-d",
        source_name="PubMed",
        source_location="vitamin d insulin resistance adults",
        text="sample abstract",
        metadata={"uid": "12345678"},
    )

    assert _document_dedupe_key(document) == "pubmed:12345678"


def test_document_dedupe_key_falls_back_to_document_id() -> None:
    document = RawDocument(
        document_id="doc-2",
        source_id="pubmed-topic-fiber",
        source_name="PubMed",
        source_location="dietary fiber metabolic syndrome adults",
        text="sample abstract",
        metadata={},
    )

    assert _document_dedupe_key(document) == "doc-2"

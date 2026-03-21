from __future__ import annotations

import csv
import hashlib
import json
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen

from nutritional_rag.etl.models import ExtractionSource, RawDocument


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        chunk = data.strip()
        if chunk:
            self._chunks.append(chunk)

    def get_text(self) -> str:
        return "\n".join(self._chunks)


def _read_location(location: str) -> bytes:
    if location.startswith(("http://", "https://")):
        with urlopen(location, timeout=30) as response:
            return response.read()
    return Path(location).read_bytes()


def _stable_document_id(source_id: str, index: int, text: str) -> str:
    payload = f"{source_id}:{index}:{text}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:20]


def _clean_row_text(row: dict[str, object]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in row.items() if value not in (None, ""))


def extract_source(source: ExtractionSource) -> list[RawDocument]:
    if source.kind == "csv":
        return _extract_csv(source)
    if source.kind == "json":
        return _extract_json(source)
    if source.kind == "html":
        return _extract_html(source)
    if source.kind == "text":
        return _extract_text(source)
    if source.kind == "pdf":
        return _extract_pdf(source)
    if source.kind == "pubmed":
        return _extract_pubmed(source)

    raise ValueError(f"Unsupported source kind: {source.kind}")


def _extract_csv(source: ExtractionSource) -> list[RawDocument]:
    content = _read_location(source.location).decode("utf-8-sig", errors="ignore")
    reader = csv.DictReader(content.splitlines())
    documents: list[RawDocument] = []

    for index, row in enumerate(reader):
        text = _clean_row_text(dict(row))
        if not text:
            continue

        title = row.get("description") or row.get("name") or row.get("food")
        document_id = _stable_document_id(source.source_id, index, text)
        metadata = {"row_index": index, **source.metadata}

        documents.append(
            RawDocument(
                document_id=document_id,
                source_id=source.source_id,
                source_name=source.source_name,
                source_location=source.location,
                title=str(title) if title else None,
                text=text,
                metadata=metadata,
            )
        )

    return documents


def _extract_json(source: ExtractionSource) -> list[RawDocument]:
    payload = json.loads(_read_location(source.location).decode("utf-8", errors="ignore"))
    records: list[dict[str, object]]

    if isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        records = [{"value": payload}]

    documents: list[RawDocument] = []
    for index, record in enumerate(records):
        text = _clean_row_text(record)
        if not text:
            continue

        title = record.get("description") or record.get("name") or record.get("title")
        document_id = _stable_document_id(source.source_id, index, text)
        metadata = {"record_index": index, **source.metadata}

        documents.append(
            RawDocument(
                document_id=document_id,
                source_id=source.source_id,
                source_name=source.source_name,
                source_location=source.location,
                title=str(title) if title else None,
                text=text,
                metadata=metadata,
            )
        )

    return documents


def _extract_html(source: ExtractionSource) -> list[RawDocument]:
    html = _read_location(source.location).decode("utf-8", errors="ignore")
    parser = _HTMLTextExtractor()
    parser.feed(html)
    text = parser.get_text()

    if not text:
        return []

    document_id = _stable_document_id(source.source_id, 0, text)
    return [
        RawDocument(
            document_id=document_id,
            source_id=source.source_id,
            source_name=source.source_name,
            source_location=source.location,
            title=source.metadata.get("title") if source.metadata else None,
            text=text,
            metadata=dict(source.metadata),
        )
    ]


def _extract_text(source: ExtractionSource) -> list[RawDocument]:
    text = _read_location(source.location).decode("utf-8", errors="ignore").strip()
    if not text:
        return []

    document_id = _stable_document_id(source.source_id, 0, text)
    return [
        RawDocument(
            document_id=document_id,
            source_id=source.source_id,
            source_name=source.source_name,
            source_location=source.location,
            title=source.metadata.get("title") if source.metadata else None,
            text=text,
            metadata=dict(source.metadata),
        )
    ]


def _extract_pdf(source: ExtractionSource) -> list[RawDocument]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PDF extraction requires 'langchain-community' and 'pypdf'. "
            "Install with: pip install langchain-community pypdf"
        ) from error

    loader = PyPDFLoader(source.location)
    lc_pages = loader.load()
    page_count = len(lc_pages)
    base_title = source.metadata.get("title") if source.metadata else None

    documents: list[RawDocument] = []
    for lc_doc in lc_pages:
        text = lc_doc.page_content.strip()
        if not text:
            continue

        # LangChain stores 0-based page index in metadata["page"]
        page_number = lc_doc.metadata.get("page", 0) + 1
        document_id = _stable_document_id(source.source_id, page_number - 1, text)
        metadata = {
            "page_number": page_number,
            "page_count": page_count,
            **source.metadata,
        }

        documents.append(
            RawDocument(
                document_id=document_id,
                source_id=source.source_id,
                source_name=source.source_name,
                source_location=source.location,
                title=base_title,
                text=text,
                metadata=metadata,
            )
        )

    return documents


def _extract_pubmed(source: ExtractionSource) -> list[RawDocument]:
    try:
        from langchain_community.document_loaders import PubMedLoader
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PubMed extraction requires 'langchain-community'. "
            "Install with: pip install langchain-community"
        ) from error

    query = source.location.strip()
    if not query:
        return []

    raw_max_docs = source.metadata.get("load_max_docs", 10)
    try:
        load_max_docs = max(1, int(raw_max_docs))
    except (TypeError, ValueError):
        load_max_docs = 10

    try:
        loader = PubMedLoader(query=query, load_max_docs=load_max_docs)
        lc_docs = loader.load()
    except ImportError as error:
        raise ModuleNotFoundError(
            "PubMed extraction requires 'xmltodict'. Install with: pip install xmltodict"
        ) from error
    base_metadata = dict(source.metadata)
    base_metadata.pop("load_max_docs", None)

    documents: list[RawDocument] = []
    for index, lc_doc in enumerate(lc_docs):
        text = (lc_doc.page_content or "").strip()
        if not text:
            continue

        lc_metadata = lc_doc.metadata or {}
        title = _extract_pubmed_title(lc_metadata)
        metadata = {
            "query": query,
            "result_index": index,
            **base_metadata,
            **lc_metadata,
        }
        document_id = _stable_document_id(source.source_id, index, text)

        documents.append(
            RawDocument(
                document_id=document_id,
                source_id=source.source_id,
                source_name=source.source_name,
                source_location=source.location,
                title=str(title) if title else None,
                text=text,
                metadata=metadata,
            )
        )

    return documents


def _extract_pubmed_title(metadata: dict[str, object]) -> str | None:
    title = metadata.get("Title") or metadata.get("title")
    if title is None:
        return None
    if isinstance(title, str):
        return title.strip() or None
    if isinstance(title, dict):
        text = title.get("#text")
        prefix = title.get("i")
        if isinstance(prefix, str) and isinstance(text, str):
            combined = f"{prefix} {text}".strip()
            return combined or None
        if isinstance(text, str):
            return text.strip() or None
    return str(title)

from __future__ import annotations

import re
from typing import Iterable

from nutritional_rag.etl.models import RawDocument, TransformedDocument

_NUMERIC_VALUE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)")

_CANONICAL_NUTRIENT_MAP = {
    "protein": "protein_g",
    "protein_g": "protein_g",
    "carbs": "carbs_g",
    "carbohydrates": "carbs_g",
    "carbs_g": "carbs_g",
    "fat": "fat_g",
    "fat_g": "fat_g",
    "calories": "calories_kcal",
    "kcal": "calories_kcal",
    "calories_kcal": "calories_kcal",
    "fiber": "fiber_g",
    "fiber_g": "fiber_g",
    "sodium": "sodium_mg",
    "sodium_mg": "sodium_mg",
}

_NUTRITION_KEYWORDS = {
    "nutrition",
    "nutrient",
    "protein",
    "carbohydrate",
    "carbohydrates",
    "carbs",
    "fat",
    "fiber",
    "calories",
    "kcal",
    "micronutrient",
    "macronutrient",
    "diet",
    "meal",
    "meals",
    "hydration",
    "electrolyte",
    "sodium",
    "potassium",
    "supplement",
}

_NON_NUTRITION_KEYWORDS = {
    "workout",
    "training",
    "sets",
    "reps",
    "repetition",
    "repetitions",
    "bench",
    "squat",
    "deadlift",
    "cardio",
    "program",
    "routine",
    "periodization",
}


def _normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    non_empty = [line for line in lines if line]
    return "\n".join(non_empty)


def _extract_numeric_value(value: str) -> float | None:
    match = _NUMERIC_VALUE_PATTERN.search(value)
    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def _canonical_key(key: str) -> str | None:
    normalized = key.strip().lower().replace(" ", "_")
    return _CANONICAL_NUTRIENT_MAP.get(normalized)


def _extract_nutrients_from_lines(lines: Iterable[str]) -> dict[str, float]:
    nutrients: dict[str, float] = {}

    for line in lines:
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        canonical = _canonical_key(key)
        if not canonical:
            continue

        parsed_value = _extract_numeric_value(value)
        if parsed_value is None:
            continue

        nutrients[canonical] = parsed_value

    return nutrients


def _keyword_hits(text: str, keywords: set[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(keyword) for keyword in keywords)


def _nutrition_score(text: str) -> tuple[int, int, int]:
    positive_hits = _keyword_hits(text, _NUTRITION_KEYWORDS)
    negative_hits = _keyword_hits(text, _NON_NUTRITION_KEYWORDS)
    score = positive_hits - negative_hits
    return score, positive_hits, negative_hits


def transform_document(document: RawDocument) -> TransformedDocument:
    clean_text = _normalize_whitespace(document.text)
    nutrients = _extract_nutrients_from_lines(clean_text.splitlines())
    score, positive_hits, negative_hits = _nutrition_score(clean_text)

    metadata = {
        **document.metadata,
        "has_nutrients": bool(nutrients),
        "nutrition_score": score,
        "nutrition_keyword_hits": positive_hits,
        "non_nutrition_keyword_hits": negative_hits,
        "is_nutrition_content": score >= 2 or bool(nutrients),
    }

    return TransformedDocument(
        document_id=document.document_id,
        source_id=document.source_id,
        title=document.title,
        clean_text=clean_text,
        nutrient_values=nutrients,
        metadata=metadata,
    )

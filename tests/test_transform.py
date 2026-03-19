from nutritional_rag.etl.models import RawDocument
from nutritional_rag.etl.transform import transform_document


def test_transform_document_extracts_nutrients() -> None:
    raw = RawDocument(
        document_id="doc-1",
        source_id="foods",
        source_name="foods",
        source_location="local",
        title="Greek yogurt",
        text="name: Greek yogurt\nprotein_g: 17\ncarbs_g: 6\nfat_g: 0\ncalories_kcal: 100",
    )

    transformed = transform_document(raw)

    assert transformed.nutrient_values["protein_g"] == 17
    assert transformed.nutrient_values["carbs_g"] == 6
    assert transformed.nutrient_values["fat_g"] == 0
    assert transformed.nutrient_values["calories_kcal"] == 100
    assert transformed.metadata["has_nutrients"] is True
    assert transformed.metadata["is_nutrition_content"] is True
    assert transformed.metadata["nutrition_score"] >= 2

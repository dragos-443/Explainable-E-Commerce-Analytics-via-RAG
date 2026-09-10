from ecommerce_rag.cloud.validation import compare_manifests


def _manifest(row_count: int = 10, field_type: str = "string") -> dict:
    schema = {
        "type": "struct",
        "fields": [{"name": "id", "type": field_type, "nullable": False, "metadata": {}}],
    }
    return {
        "spark_version": "3.5.5",
        "datasets": [
            {"layer": "curated", "dataset": "example", "row_count": row_count, "schema": schema}
        ],
        "phase1_contracts": [
            {"layer": "curated", "dataset": "example", "schema": schema}
        ],
    }


def test_equal_manifests_are_compatible() -> None:
    assert compare_manifests(_manifest(), _manifest())["compatible"] is True


def test_row_count_difference_is_rejected() -> None:
    result = compare_manifests(_manifest(), _manifest(row_count=11))
    assert result["compatible"] is False
    assert result["datasets"][0]["row_count_equal"] is False


def test_schema_difference_is_rejected() -> None:
    result = compare_manifests(_manifest(), _manifest(field_type="long"))
    assert result["compatible"] is False
    assert result["contract_schemas_match"] is False

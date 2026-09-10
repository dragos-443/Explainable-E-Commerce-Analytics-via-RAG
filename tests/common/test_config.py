from pathlib import Path

from ecommerce_rag.common.config import load_config


def test_config_directory_can_be_supplied_by_environment(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "aws.yml").write_text(
        "environment: aws\nstorage:\n  scheme: s3a\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ECOMMERCE_RAG_CONFIG_DIR", str(tmp_path))
    assert load_config("aws")["storage"]["scheme"] == "s3a"

from pathlib import Path


ROOT = Path("/workspace")


def test_streamlit_application_is_packaged_for_compose_environment():
    requirements = (ROOT / "requirements-container.txt").read_text(encoding="utf-8")
    compose = (ROOT / "compose.yml").read_text(encoding="utf-8")
    launcher = (ROOT / "scripts/rag/start_ui.ps1").read_text(encoding="utf-8")

    assert "streamlit==" in requirements
    assert "${STREAMLIT_PORT:-8501}:8501" in compose
    assert "streamlit run" in launcher
    assert "_stcore/health" in launcher


def test_streamlit_source_exposes_all_required_sections():
    source = (
        ROOT / "src/ecommerce_rag/app/streamlit_app.py"
    ).read_text(encoding="utf-8")

    for required in (
        "Domanda in italiano",
        "Metriche osservate",
        "Confronto e variazione",
        "Spiegazione generata",
        "Recensioni di supporto",
        "Testo originale",
        "Traduzione automatica in italiano",
    ):
        assert required in source

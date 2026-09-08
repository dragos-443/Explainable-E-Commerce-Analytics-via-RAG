from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = Path("/workspace/src/ecommerce_rag/app/streamlit_app.py")
RESULT_HARNESS = Path("/workspace/tests/app/phase8_result_harness.py")


def test_streamlit_initial_view_loads_without_starting_the_heavy_pipeline():
    app = AppTest.from_file(str(APP)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "Explainable E-Commerce Analytics via RAG"
    assert app.markdown[0].value == (
        "Fai una domanda sui dati e approfondisci i risultati attraverso le "
        "recensioni dei clienti."
    )
    assert app.text_area[0].label == "Domanda in italiano"
    assert {button.label for button in app.button} == {
        "Analizza e spiega",
        "Carica domanda di esempio",
    }


def test_streamlit_renders_a_real_end_to_end_result():
    app = AppTest.from_file(str(RESULT_HARNESS)).run(timeout=30)

    assert not app.exception
    assert [item.value for item in app.subheader] == [
        "Domanda analizzata",
        "Metriche osservate",
        "Confronto e variazione",
        "Spiegazione generata",
        "Confronto dei temi segnalati",
        "Recensioni di supporto",
    ]
    assert len(app.metric) == 4
    assert any("Recensione 1 · ⭐ 1/5" == item.label for item in app.expander)
    assert any(
        item.value.startswith("**ID recensione:** `") for item in app.markdown
    )
    rendered = [item.value for item in app.markdown]
    assert all(
        not item.value.startswith("Lingua associata al dataset:")
        for item in app.caption
    )
    assert any(
        item.value.startswith("Ambito analizzato:") for item in app.caption
    )
    assert all(not item.value.startswith("Intento:") for item in app.caption)
    assert any(
        item.value
        == (
            "Sono mostrate alcune recensioni pertinenti alla domanda, non tutte le "
            "opinioni dei clienti."
        )
        for item in app.caption
    )
    assert any(item.startswith("**Spark rileva che**, ") for item in rendered)
    assert any(
        item.startswith("**Nell’intera popolazione analizzata**, ")
        for item in rendered
    )
    assert any(
        item.startswith("**Le recensioni recuperate descrivono**")
        for item in rendered
    )
    assert "**Argomenti ricavati dalle recensioni**" not in rendered
    assert all(
        item.label != "Riferimenti usati nella sintesi" for item in app.expander
    )
    assert all(
        "sintesi deterministica di sicurezza" not in item.value.lower()
        for item in app.info
    )

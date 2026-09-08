from ecommerce_rag.app.presentation import (
    analysis_scope,
    comparison_rows,
    metric_cards,
    query_identifier,
    rating_display,
    review_view,
    theme_rows,
)
from ecommerce_rag.app import streamlit_app


def _result():
    return {
        "analytics": {
            "metrics": {
                "average_rating": {"value": 3.5, "baseline_value": 4.0, "change": -0.5},
                "negative_review_rate": {"value": 0.2, "baseline_value": 0.1, "change": 0.1},
                "late_delivery_rate": {"value": 0.15, "baseline_value": 0.1, "change": 0.05},
                "order_volume": {"value": 100, "baseline_value": 90, "change": None},
            }
        },
        "theme_evidence": {
            "ranked_hypotheses": [
                {
                    "theme": "delivery_delay",
                    "current_mentions": 12,
                    "current_mention_rate": 0.3,
                    "mention_rate_change": 0.1,
                }
            ]
        },
    }


def test_metric_and_comparison_views_format_values_for_italian_ui():
    cards = metric_cards(_result())

    assert cards[0]["value"] == "3.500"
    assert cards[1]["value"] == "20.00%"
    assert cards[2]["delta"] == "+5.00%"
    assert comparison_rows(_result())[3]["Variazione"] == "n.d."


def test_theme_view_uses_readable_label_and_population_rate():
    row = theme_rows(_result())[0]

    assert row["Tema"] == "Ritardo nella consegna"
    assert row["Quota nel gruppo"] == "30.00%"
    assert "Variazione vs baseline" not in row


def test_theme_view_shows_variation_only_with_temporal_baseline():
    result = _result()
    result["analytics"]["baseline"] = {
        "filters": {"start_month": "2017-12", "end_month": "2018-02"}
    }

    row = theme_rows(result)[0]

    assert row["Variazione vs baseline"] == "+10.00%"


def test_rating_display_combines_stars_and_unambiguous_numeric_score():
    assert rating_display(1) == "⭐ 1/5"
    assert rating_display(4) == "⭐ 4/5"
    assert rating_display(None) == "Valutazione n.d."


def test_single_theme_table_is_hidden_because_narrative_already_contains_it(
    monkeypatch,
):
    recorder = _StreamlitRecorder()
    monkeypatch.setattr(streamlit_app, "st", recorder)

    streamlit_app._render_themes(_result())

    assert recorder.subheaders == []
    assert recorder.tables == []


def test_review_view_keeps_original_when_translation_fails():
    review = review_view(
        {
            "review_id": "r1",
            "document_original": "Não recebi.",
            "retrieved_for_theme": "non_delivery",
            "metadata": {"review_language": "pt", "review_score": 1},
            "translation": {
                "status": "failed_original_available",
                "error": "offline",
                "translation_glossary_version": "pt-it-ecommerce-v1",
                "glossary_corrections": [],
            },
        }
    )

    assert review["original"] == "Não recebi."
    assert review["translation"] is None
    assert review["translation_error"] == "offline"


def test_review_view_exposes_applied_translation_glossary_rules():
    review = review_view(
        {
            "review_id": "r1",
            "document_original": "O cobre leito parece um lençol",
            "metadata": {},
            "translation": {
                "status": "translated",
                "message": "Il copriletto sembra un lenzuolo",
                "translation_glossary_version": "pt-it-ecommerce-v1",
                "glossary_corrections": [
                    {
                        "rule": "cobre_leito",
                        "source_term": "cobre-leito / cobre leito",
                        "replacement_it": "copriletto",
                    }
                ],
            },
        }
    )

    assert review["translation"] == "Il copriletto sembra un lenzuolo"
    assert review["glossary_corrections"][0]["rule"] == "cobre_leito"


def test_query_identifier_is_stable_and_changes_with_filters():
    first = query_identifier("Perché?", {"category": None})

    assert first == query_identifier("Perché?", {"category": None})
    assert first != query_identifier("Perché?", {"category": "office_furniture"})


def test_analysis_scope_hides_intent_and_uses_readable_italian_labels():
    scope = analysis_scope(
        {
            "intent": "analyze_low_rating_complaints",
            "category": "bed_bath_table",
            "requested_theme": "quality_or_expectation",
            "customer_state": None,
            "start_month": None,
            "end_month": None,
        }
    )

    assert scope == "Letto, bagno e tavola · Qualità inferiore alle aspettative"
    assert "analyze_low_rating_complaints" not in scope


def test_analysis_scope_formats_state_and_single_month():
    scope = analysis_scope(
        {
            "category": "office_furniture",
            "customer_state": "SP",
            "start_month": "2018-03",
            "end_month": "2018-03",
        }
    )

    assert scope == "Mobili per ufficio · Stato: SP · Marzo 2018"


class _MetricColumn:
    def metric(self, label, value, delta):
        pass


class _StreamlitRecorder:
    def __init__(self):
        self.subheaders = []
        self.tables = []
        self.captions = []

    def subheader(self, value):
        self.subheaders.append(value)

    def columns(self, count):
        return [_MetricColumn() for _ in range(count)]

    def table(self, value):
        self.tables.append(value)

    def caption(self, value):
        self.captions.append(value)


def test_metrics_hide_comparison_without_temporal_baseline(monkeypatch):
    result = _result()
    result["analytics"]["baseline"] = {"filters": None}
    recorder = _StreamlitRecorder()
    monkeypatch.setattr(streamlit_app, "st", recorder)

    streamlit_app._render_metrics(result)

    assert "Confronto e variazione" not in recorder.subheaders
    assert recorder.tables == []
    assert recorder.captions == [
        "Metriche calcolate sull’intero periodo disponibile."
    ]


def test_metrics_show_comparison_with_temporal_baseline(monkeypatch):
    result = _result()
    result["analytics"]["baseline"] = {
        "filters": {"start_month": "2017-12", "end_month": "2018-02"}
    }
    recorder = _StreamlitRecorder()
    monkeypatch.setattr(streamlit_app, "st", recorder)

    streamlit_app._render_metrics(result)

    assert "Confronto e variazione" in recorder.subheaders
    assert len(recorder.tables) == 1
    assert "2017-12 – 2018-02" in recorder.captions[0]


def test_loading_example_clears_all_manual_filter_overrides():
    class SessionState:
        question_input = "domanda precedente"
        category_input = "office_furniture"
        state_input = "SP"
        start_month_input = "2018-01"
        end_month_input = "2018-03"

    selected = type("Example", (), {"question": "Nuova domanda automatica"})()
    state = SessionState()

    streamlit_app._load_example(state, selected)

    assert state.question_input == "Nuova domanda automatica"
    assert state.category_input == ""
    assert state.state_input == ""
    assert state.start_month_input == ""
    assert state.end_month_input == ""

"""Simple end-to-end Streamlit application for explainable Olist analytics."""

from __future__ import annotations

import os

import streamlit as st

from ecommerce_rag.app.demo_cases import APP_EXAMPLE_CASES
from ecommerce_rag.app.presentation import (
    analysis_scope,
    comparison_rows,
    metric_cards,
    query_identifier,
    rating_display,
    review_view,
    theme_rows,
)
from ecommerce_rag.app.runtime import build_application_runtime
from ecommerce_rag.rag.question_interpreter import interpret_question


@st.cache_resource(show_spinner=False)
def application_runtime():
    return build_application_runtime(os.getenv("APP_ENV", "local"))


def _load_example(session_state, selected_case) -> None:
    """Load only the natural-language question and clear every manual override."""
    session_state.question_input = selected_case.question
    session_state.category_input = ""
    session_state.state_input = ""
    session_state.start_month_input = ""
    session_state.end_month_input = ""


def _render_metrics(result: dict) -> None:
    st.subheader("Metriche osservate")
    columns = st.columns(4)
    for column, card in zip(columns, metric_cards(result)):
        column.metric(card["label"], card["value"], card["delta"])
    baseline = result["analytics"]["baseline"]["filters"]
    if baseline:
        st.subheader("Confronto e variazione")
        st.table(comparison_rows(result))
        st.caption(
            "La baseline usa i tre mesi precedenti al periodo selezionato: "
            "{} – {}.".format(baseline["start_month"], baseline["end_month"])
        )
    else:
        st.caption("Metriche calcolate sull’intero periodo disponibile.")


def _render_explanation(result: dict) -> None:
    st.subheader("Spiegazione generata")
    insufficient = result["insufficient_evidence"]
    generation = result.get("generation")
    if insufficient["is_insufficient"]:
        st.warning("Evidenza insufficiente: {}".format(insufficient["reason"]))
        return
    if generation is None:
        st.warning("La spiegazione non è disponibile.")
        return
    quantitative = generation.get("quantitative_summary")
    thematic = generation.get("theme_summary")
    if quantitative:
        st.markdown("**Spark rileva che**, {}".format(
            quantitative[:1].lower() + quantitative[1:]
        ))
    if thematic:
        st.markdown("**Nell’intera popolazione analizzata**, {}".format(
            thematic[:1].lower() + thematic[1:]
        ))
    interpretation = generation["interpretation"]
    review_lead = "Le recensioni recuperate descrivono"
    if interpretation.startswith(review_lead):
        interpretation = interpretation.replace(
            review_lead, "**{}**".format(review_lead), 1
        )
    st.markdown(interpretation)


def _render_themes(result: dict) -> None:
    rows = theme_rows(result)
    # A single theme is already quantified in the narrative explanation.
    # The table adds value only when it permits a comparison between themes.
    if len(rows) < 2:
        return
    st.subheader("Confronto dei temi segnalati")
    st.table(rows)
    population = result["theme_evidence"]["current_population"]
    st.caption(
        "Le quote sono calcolate da Spark sull’intero gruppo di {} recensioni "
        "negative con testo. I temi sono multi-label e costituiscono ipotesi, "
        "non cause dimostrate.".format(population["negative_text_reviews"])
    )


def _render_reviews(result: dict) -> None:
    st.subheader("Recensioni di supporto")
    evidence = result["context"]["review_evidence"]
    if not evidence:
        st.info("Nessuna recensione compatibile con i filtri e i temi disponibili.")
        return
    st.caption(
        "Sono mostrate alcune recensioni pertinenti alla domanda, non tutte le "
        "opinioni dei clienti."
    )
    for index, item in enumerate(evidence, start=1):
        review = review_view(item)
        with st.expander(
            "Recensione {} · {}".format(
                index,
                rating_display(review["review_score"]),
            ),
            expanded=index == 1,
        ):
            st.markdown("**ID recensione:** `{}`".format(review["review_id"]))
            st.markdown("**Testo originale**")
            st.write(review["original"])
            st.markdown("**Traduzione automatica in italiano**")
            if review["translation"]:
                st.write(review["translation"])
                if review["glossary_corrections"]:
                    applied = ", ".join(
                        "{} → {}".format(
                            item["source_term"], item["replacement_it"]
                        )
                        for item in review["glossary_corrections"]
                    )
                    st.caption(
                        "Correzione automatica del glossario {}: {}.".format(
                            review["translation_glossary_version"], applied
                        )
                    )
            else:
                st.warning(
                    "Traduzione non disponibile; il testo originale rimane consultabile."
                )
                if review["translation_error"]:
                    st.caption("Errore: {}".format(review["translation_error"]))
            metadata = {
                "Tema di recupero": review["retrieved_for_theme"] or "n.d.",
                "Categoria": review["product_category"] or "n.d.",
                "Stato cliente": review["customer_state"] or "n.d.",
                "Mese di acquisto": review["purchase_month"] or "n.d.",
            }
            st.json(metadata, expanded=False)


def render_result(result: dict) -> None:
    st.divider()
    st.subheader("Domanda analizzata")
    st.write(result["question"]["question_original"])
    st.caption("Ambito analizzato: {}".format(analysis_scope(result["question"])))
    _render_metrics(result)
    _render_explanation(result)
    _render_themes(result)
    _render_reviews(result)


def main() -> None:
    st.set_page_config(
        page_title="Explainable E-Commerce Analytics",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Explainable E-Commerce Analytics via RAG")
    st.write(
        "Fai una domanda sui dati e approfondisci i risultati attraverso le "
        "recensioni dei clienti."
    )

    if "question_input" not in st.session_state:
        st.session_state.question_input = APP_EXAMPLE_CASES[0].question
    with st.sidebar:
        st.header("Esempi")
        selected_case = st.selectbox(
            "Scenario dimostrativo",
            APP_EXAMPLE_CASES,
            format_func=lambda case: case.title,
        )
        if st.button("Carica domanda di esempio", use_container_width=True):
            _load_example(st.session_state, selected_case)
        st.info(
            "L’elaborazione locale può richiedere alcuni minuti. Le traduzioni "
            "già presenti nella cache sono più rapide."
        )

    with st.form("question_form"):
        question_text = st.text_area(
            "Domanda in italiano",
            key="question_input",
            height=100,
            placeholder="Perché il rating è diminuito a marzo 2018?",
        )
        with st.expander("Filtri opzionali"):
            category = st.text_input(
                "Categoria normalizzata", key="category_input", placeholder="office_furniture"
            )
            state = st.text_input(
                "Stato cliente", key="state_input", placeholder="SP", max_chars=2
            )
            period_columns = st.columns(2)
            start_month = period_columns[0].text_input(
                "Mese iniziale", key="start_month_input", placeholder="2018-03"
            )
            end_month = period_columns[1].text_input(
                "Mese finale", key="end_month_input", placeholder="2018-03"
            )
        submitted = st.form_submit_button(
            "Analizza e spiega", type="primary", use_container_width=True
        )

    if submitted:
        st.session_state.last_result = None
        if not question_text.strip():
            st.warning("Inserisci una domanda prima di avviare l’analisi.")
        else:
            try:
                interpreted = interpret_question(
                    question_text,
                    category=category.strip() or None,
                    customer_state=state.strip() or None,
                    start_month=start_month.strip() or None,
                    end_month=end_month.strip() or None,
                )
                filters = {
                    "category": interpreted.category,
                    "customer_state": interpreted.customer_state,
                    "start_month": interpreted.start_month,
                    "end_month": interpreted.end_month,
                }
                with st.spinner("Spark, Chroma e LLM stanno elaborando la domanda…"):
                    result = application_runtime().pipeline.run(
                        interpreted,
                        query_id=query_identifier(question_text, filters),
                    )
                st.session_state.last_result = result
            except ValueError as error:
                st.error("Domanda o filtri non validi: {}".format(error))
            except Exception as error:
                st.error(
                    "L’analisi non è stata completata. Verifica che Spark, HDFS e "
                    "Chroma siano attivi e riprova."
                )
                with st.expander("Dettaglio tecnico"):
                    st.code("{}: {}".format(type(error).__name__, error))

    if st.session_state.get("last_result"):
        render_result(st.session_state.last_result)


if __name__ == "__main__":
    main()

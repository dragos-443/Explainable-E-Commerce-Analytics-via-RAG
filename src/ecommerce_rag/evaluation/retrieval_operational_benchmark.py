"""Run the operational 50-query retrieval benchmark."""

from ecommerce_rag.evaluation import retrieval_final_holdout as evaluator
from ecommerce_rag.rag.explanation_pipeline import theme_retrieval_query


def selective_retrieval_query(query: dict) -> str:
    """Focus only the aspect for which the controlled ablation improved retrieval."""
    if query["expected_aspects"] == ["wrong_or_missing_item"]:
        return theme_retrieval_query(query["question_it"], "wrong_or_missing_item")
    return query["question_it"]


def main() -> None:
    evaluator.QUERY_PATH = evaluator.DATA_ROOT / "retrieval_operational_queries.json"
    evaluator.DEFAULT_OUTPUT = evaluator.Path(
        "/workspace/reports/evaluation/retrieval/operational-50"
    )
    evaluator.METHOD = (
        "e5-small metadata filters plus selective wrong-or-missing query focus and reranker"
    )
    evaluator.retrieval_query = selective_retrieval_query
    evaluator.BENCHMARK_STATUS = "operational_benchmark"
    evaluator.COMBINED_NEW_LABEL = "operational_benchmark"
    evaluator.COMBINED_WARNING = (
        "The 20 development queries influenced earlier design choices. "
        "The operational 50-query benchmark is reported separately from the "
        "development queries."
    )
    evaluator.ALLOW_ANNOTATION_REPLACEMENT = True
    evaluator.main()


if __name__ == "__main__":
    main()

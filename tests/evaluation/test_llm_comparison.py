from ecommerce_rag.evaluation.llm_comparison import (
    estimate_terra_cost_usd,
    normalized_usage,
    summarize_run,
)


def test_terra_cost_distinguishes_cached_input_tokens() -> None:
    usage = {
        "input_tokens": 1_000_000,
        "input_tokens_details": {"cached_tokens": 250_000},
        "output_tokens": 100_000,
    }

    assert normalized_usage(usage) == {
        "input_tokens": 1_000_000,
        "cached_input_tokens": 250_000,
        "output_tokens": 100_000,
    }
    assert estimate_terra_cost_usd(usage) == 2.75


def test_summary_keeps_validated_generation_and_fallback_separate() -> None:
    cases = [
        {
            "verification_passed": True,
            "generation_status": "llm_generated_validated",
            "generation_seconds": 2.0,
            "end_to_end_seconds": 5.0,
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 0,
                "output_tokens": 20,
            },
            "estimated_api_cost_usd": 0.001,
        },
        {
            "verification_passed": True,
            "generation_status": "validated_fallback",
            "generation_seconds": 4.0,
            "end_to_end_seconds": 7.0,
            "usage": {
                "input_tokens": 90,
                "cached_input_tokens": 10,
                "output_tokens": 10,
            },
            "estimated_api_cost_usd": 0.002,
        },
        {
            "verification_passed": True,
            "generation_status": None,
            "generation_seconds": 0.0,
            "end_to_end_seconds": 1.0,
            "usage": {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "output_tokens": 0,
            },
            "estimated_api_cost_usd": 0.0,
        },
    ]

    summary = summarize_run("openai", cases)

    assert summary["verification_passed"] == 3
    assert summary["llm_generated_validated"] == 1
    assert summary["validated_fallback"] == 1
    assert summary["insufficient_evidence"] == 1
    assert summary["mean_generation_seconds"] == 3.0
    assert summary["usage"]["input_tokens"] == 190
    assert summary["estimated_api_cost_usd"] == 0.003

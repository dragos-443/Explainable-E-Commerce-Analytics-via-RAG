from ecommerce_rag.app.demo_cases import APP_EXAMPLE_CASES, DEMO_CASES, THEMATIC_EXAMPLE_CASES
from ecommerce_rag.rag.question_interpreter import interpret_question


def test_demo_suite_covers_distinct_required_scenarios():
    assert len(DEMO_CASES) >= 3
    assert len({case.case_id for case in DEMO_CASES}) == len(DEMO_CASES)
    assert {case.kind for case in DEMO_CASES} == {
        "logistics",
        "product_issues",
        "review_analysis",
        "insufficient_evidence",
    }


def test_demo_questions_are_interpreted_as_specified():
    for case in DEMO_CASES:
        interpreted = interpret_question(
            case.question,
            category=case.category,
            start_month=case.start_month,
            end_month=case.end_month,
        )

        assert interpreted.intent == case.expected_intent
        assert interpreted.category == case.category
        assert interpreted.start_month == case.start_month
        assert interpreted.end_month == case.end_month


def test_streamlit_examples_add_distinct_non_logistics_themes():
    assert len(APP_EXAMPLE_CASES) == len(DEMO_CASES) + len(THEMATIC_EXAMPLE_CASES)
    interpreted = [
        interpret_question(case.question, category=case.category)
        for case in THEMATIC_EXAMPLE_CASES
    ]

    assert {item.requested_theme for item in interpreted} == {
        "quality_or_expectation",
        "damaged_or_defective",
        "wrong_or_missing_item",
        "service_or_refund",
    }
    assert not {
        "non_delivery",
        "delivery_delay",
    }.intersection(item.requested_theme for item in interpreted)

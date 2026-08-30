from ecommerce_rag.app.demo_cases import DEMO_CASES
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

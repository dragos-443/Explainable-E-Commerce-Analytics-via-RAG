import json
from pathlib import Path

from ecommerce_rag.app.streamlit_app import render_result
from ecommerce_rag.rag.context_builder.builder import build_grounded_context
from ecommerce_rag.rag.explanation_pipeline import _safe_fallback


RESULT = Path("/workspace/reports/demo/phase6/logistics_march_2018.json")
payload = json.loads(RESULT.read_text(encoding="utf-8"))
context = build_grounded_context(
    payload["question"],
    payload["analytics"],
    payload["theme_evidence"],
    payload["context"]["review_evidence"],
    3,
)
payload["context"] = context
payload["generation"] = _safe_fallback(context, "test harness")
render_result(payload)

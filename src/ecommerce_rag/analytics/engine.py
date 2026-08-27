"""Reusable Spark analytics engine for the curated Olist datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from decimal import Decimal
from typing import Any, Dict, Optional, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ecommerce_rag.analytics.metrics import (
    BASELINE_MONTH_COUNT,
    KPI_DEFINITIONS,
    LOW_RATING_SCORES,
    METRIC_DEFINITION_VERSION,
    negative_indicator,
)


@dataclass(frozen=True)
class AnalyticsFilters:
    """Filters supported consistently across review and order populations."""

    product_category: Optional[str] = None
    customer_state: Optional[str] = None
    start_month: Optional[str] = None
    end_month: Optional[str] = None

    def validated(self) -> "AnalyticsFilters":
        start = parse_month(self.start_month) if self.start_month else None
        end = parse_month(self.end_month) if self.end_month else start
        if end and not start:
            raise ValueError("end_month requires start_month")
        if start and end and start > end:
            raise ValueError("start_month must not be after end_month")
        return replace(
            self,
            product_category=self.product_category or None,
            customer_state=(self.customer_state.upper() if self.customer_state else None),
            start_month=format_month(start) if start else None,
            end_month=format_month(end) if end else None,
        )


def parse_month(value: str) -> date:
    try:
        year, month = (int(part) for part in value.split("-"))
        return date(year, month, 1)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("Months must use YYYY-MM format: {!r}".format(value))


def format_month(value: date) -> str:
    return value.strftime("%Y-%m")


def shift_month(value: date, offset: int) -> date:
    month_index = value.year * 12 + value.month - 1 + offset
    return date(month_index // 12, month_index % 12 + 1, 1)


def baseline_filters(
    filters: AnalyticsFilters, months: int = BASELINE_MONTH_COUNT
) -> Optional[AnalyticsFilters]:
    """Return the immediately preceding fixed-size baseline window."""
    filters = filters.validated()
    if not filters.start_month:
        return None
    current_start = parse_month(filters.start_month)
    baseline_end = shift_month(current_start, -1)
    baseline_start = shift_month(current_start, -months)
    return replace(
        filters,
        start_month=format_month(baseline_start),
        end_month=format_month(baseline_end),
    )


def _month_column(frame_kind: str):
    if frame_kind == "orders":
        return F.trunc(F.col("purchase_date"), "month")
    return F.make_date(F.col("purchase_year"), F.col("purchase_month"), F.lit(1))


def apply_filters(
    frame: DataFrame, filters: AnalyticsFilters, frame_kind: str
) -> DataFrame:
    """Apply the same semantic filters to either curated population."""
    filters = filters.validated()
    if frame_kind not in {"orders", "reviews"}:
        raise ValueError("frame_kind must be 'orders' or 'reviews'")
    result = frame
    if filters.product_category:
        category_column = "order_category" if frame_kind == "orders" else "product_category"
        result = result.where(F.col(category_column) == filters.product_category)
    if filters.customer_state:
        result = result.where(F.col("customer_state") == filters.customer_state)
    if filters.start_month:
        month = _month_column(frame_kind)
        result = result.where(month >= F.lit(parse_month(filters.start_month)))
        result = result.where(month <= F.lit(parse_month(filters.end_month)))
    return result


def _native(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, date):
        return value.isoformat()
    return value


def _review_summary(reviews: DataFrame) -> Dict[str, Any]:
    row = reviews.agg(
        F.count("review_score").alias("review_count"),
        F.avg("review_score").alias("average_rating"),
        F.coalesce(F.sum(negative_indicator()), F.lit(0)).alias(
            "negative_review_count"
        ),
        *[
            F.coalesce(
                F.sum(F.when(F.col("review_score") == score, 1).otherwise(0)),
                F.lit(0),
            ).alias("rating_{}_count".format(score))
            for score in LOW_RATING_SCORES
        ],
    ).first().asDict()
    return {key: _native(value) for key, value in row.items()}


def _order_summary(orders: DataFrame) -> Dict[str, Any]:
    row = orders.agg(
        F.countDistinct("order_id").alias("order_count"),
        F.count("order_value").alias("order_value_count"),
        F.avg("order_value").alias("average_order_value"),
        F.count("is_late").alias("delivery_eligible_count"),
        F.coalesce(
            F.sum(F.when(F.col("is_late"), 1).otherwise(0)), F.lit(0)
        ).alias("late_order_count"),
        F.avg(F.col("is_late").cast("double")).alias("late_delivery_rate"),
        F.coalesce(
            F.sum(F.when(F.col("is_late"), 1).otherwise(0)), F.lit(0)
        ).alias("delivery_delay_eligible_count"),
        F.avg(F.when(F.col("is_late"), F.col("delivery_delay_days"))).alias(
            "average_delivery_delay"
        ),
    ).first().asDict()
    return {key: _native(value) for key, value in row.items()}


def _status_distribution(orders: DataFrame) -> Dict[str, Dict[str, Any]]:
    total = orders.count()
    rows = orders.groupBy("order_status").count().collect()
    return {
        (row.order_status or "unknown"): {
            "count": row["count"],
            "share": row["count"] / total if total else None,
        }
        for row in rows
    }


def _scalar_metric(
    value: Any,
    baseline_value: Any,
    numerator: Optional[int],
    denominator: Optional[int],
) -> Dict[str, Any]:
    change = None
    if value is not None and baseline_value is not None:
        change = value - baseline_value
    return {
        "value": value,
        "baseline_value": baseline_value,
        "change": change,
        "numerator": numerator,
        "denominator": denominator,
    }


class AnalyticsEngine:
    """Calculate deterministic KPIs without delegating arithmetic to an LLM."""

    def __init__(self, orders: DataFrame, reviews: DataFrame):
        self.orders = orders
        self.reviews = reviews

    def analyze(
        self, filters: Optional[AnalyticsFilters] = None, query_id: str = "analytics-query"
    ) -> Dict[str, Any]:
        selected = (filters or AnalyticsFilters()).validated()
        current_orders = apply_filters(self.orders, selected, "orders")
        current_reviews = apply_filters(self.reviews, selected, "reviews")
        current_review = _review_summary(current_reviews)
        current_order = _order_summary(current_orders)
        current_status = _status_distribution(current_orders)

        comparison = baseline_filters(selected)
        baseline_review: Dict[str, Any] = {}
        baseline_order: Dict[str, Any] = {}
        baseline_status: Dict[str, Dict[str, Any]] = {}
        if comparison:
            baseline_review = _review_summary(
                apply_filters(self.reviews, comparison, "reviews")
            )
            baseline_orders = apply_filters(self.orders, comparison, "orders")
            baseline_order = _order_summary(baseline_orders)
            baseline_status = _status_distribution(baseline_orders)

        review_count = current_review["review_count"]
        baseline_review_count = baseline_review.get("review_count")
        low_distribution = {}
        for score in LOW_RATING_SCORES:
            count = current_review["rating_{}_count".format(score)]
            baseline_count = baseline_review.get("rating_{}_count".format(score))
            share = count / review_count if review_count else None
            baseline_share = (
                baseline_count / baseline_review_count
                if baseline_review_count and baseline_count is not None
                else None
            )
            low_distribution[str(score)] = {
                "count": count,
                "share": share,
                "baseline_count": baseline_count,
                "baseline_share": baseline_share,
                "share_change": (
                    share - baseline_share
                    if share is not None and baseline_share is not None
                    else None
                ),
            }

        statuses = sorted(set(current_status) | set(baseline_status))
        status_distribution = {
            status: {
                "count": current_status.get(status, {}).get("count", 0),
                "share": current_status.get(status, {}).get(
                    "share", 0.0 if current_order["order_count"] else None
                ),
                "baseline_count": baseline_status.get(status, {}).get(
                    "count", 0 if baseline_order.get("order_count") else None
                ),
                "baseline_share": baseline_status.get(status, {}).get(
                    "share", 0.0 if baseline_order.get("order_count") else None
                ),
            }
            for status in statuses
        }

        metrics = {
            "average_rating": _scalar_metric(
                current_review["average_rating"],
                baseline_review.get("average_rating"),
                None,
                review_count,
            ),
            "rating_variation": {
                "value": (
                    current_review["average_rating"] - baseline_review["average_rating"]
                    if current_review["average_rating"] is not None
                    and baseline_review.get("average_rating") is not None
                    else None
                ),
                "baseline_required": True,
            },
            "negative_review_rate": _scalar_metric(
                (
                    current_review["negative_review_count"] / review_count
                    if review_count
                    else None
                ),
                (
                    baseline_review["negative_review_count"] / baseline_review_count
                    if baseline_review_count
                    else None
                ),
                current_review["negative_review_count"],
                review_count,
            ),
            "low_rating_distribution": low_distribution,
            "late_delivery_rate": _scalar_metric(
                current_order["late_delivery_rate"],
                baseline_order.get("late_delivery_rate"),
                current_order["late_order_count"],
                current_order["delivery_eligible_count"],
            ),
            "average_delivery_delay": _scalar_metric(
                current_order["average_delivery_delay"],
                baseline_order.get("average_delivery_delay"),
                None,
                current_order["delivery_delay_eligible_count"],
            ),
            "order_status_distribution": status_distribution,
            "order_volume": {
                "value": current_order["order_count"],
                "baseline_value": baseline_order.get("order_count"),
                "change": None,
                "numerator": current_order["order_count"],
                "denominator": current_order["order_count"],
                "comparison_note": (
                    "No direct change is computed because current and baseline "
                    "windows can cover different numbers of months."
                ),
            },
            "average_order_value": _scalar_metric(
                current_order["average_order_value"],
                baseline_order.get("average_order_value"),
                None,
                current_order["order_value_count"],
            ),
        }
        return {
            "schema_version": "1.0",
            "query_id": query_id,
            "filters": asdict(selected),
            "filter_semantics": {
                "product_category": (
                    "Orders must be strictly mono-category; review metadata must be "
                    "unambiguous across every linked order."
                ),
                "customer_state": (
                    "Review state must be unambiguous across every linked order."
                ),
                "period": (
                    "Orders use purchase_date; reviews use their unambiguous linked "
                    "purchase year and month."
                ),
            },
            "baseline": {
                "strategy": "weighted aggregate over the three months before start_month",
                "filters": asdict(comparison) if comparison else None,
            },
            "populations": {
                "reviews": review_count,
                "orders": current_order["order_count"],
                "delivery_eligible_orders": current_order["delivery_eligible_count"],
                "baseline_reviews": baseline_review_count,
                "baseline_orders": baseline_order.get("order_count"),
            },
            "metric_definition_version": METRIC_DEFINITION_VERSION,
            "metric_definitions": KPI_DEFINITIONS,
            "metrics": metrics,
            "interpretation_limit": (
                "Metrics are descriptive evidence and must not be presented as proof of causality."
            ),
        }

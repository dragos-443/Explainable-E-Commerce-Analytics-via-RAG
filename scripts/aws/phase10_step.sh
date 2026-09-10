#!/usr/bin/env bash
set -euo pipefail

action="${1:?action is required}"
bucket="${2:?bucket is required}"
run_id="${3:?run id is required}"
shift 3

root="/tmp/ecommerce-rag-phase10"
config_dir="$root/config"
report_dir="$root/reports/$action"
mkdir -p "$config_dir" "$report_dir"
aws s3 cp "s3://$bucket/artifacts/aws.yml" "$config_dir/aws.yml" --only-show-errors
aws s3 cp "s3://$bucket/artifacts/ecommerce_rag.zip" "$root/ecommerce_rag.zip" --only-show-errors
code_root="$root/python"
rm -rf "$code_root"
mkdir -p "$code_root"
aws s3 cp "s3://$bucket/artifacts/source" "$code_root" --recursive --only-show-errors
export ECOMMERCE_RAG_CONFIG_DIR="$config_dir"

pyfiles="$root/ecommerce_rag.zip"
result_root="s3://$bucket/results/$run_id/$action"
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export PYTHONPATH="$code_root${PYTHONPATH:+:$PYTHONPATH}"

upload_result() {
  status=$?
  finished_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 - "$action" "$started_at" "$finished_at" "$status" "$report_dir/step_metadata.json" <<'PY'
import json
import sys
from pathlib import Path

action, started, finished, status, destination = sys.argv[1:]
Path(destination).write_text(json.dumps({
    "action": action,
    "started_at_utc": started,
    "finished_at_utc": finished,
    "exit_code": int(status),
}, indent=2) + "\n", encoding="utf-8")
PY
  aws s3 cp "$report_dir" "$result_root" --recursive --only-show-errors || true
  exit "$status"
}
trap upload_result EXIT

cd "$code_root"
python3 -c "from ecommerce_rag.common.config import load_config; load_config('aws')" \
  2>&1 | tee "$report_dir/import_check.log"

spark_submit=(
  spark-submit
  --master yarn
  --deploy-mode client
  --archives "$pyfiles#phase10code"
  --conf spark.executorEnv.PYTHONPATH=./phase10code
)

case "$action" in
  pipeline)
    "${spark_submit[@]}" "$code_root/ecommerce_rag/preprocessing/pipeline.py" --environment aws \
      2>&1 | tee "$report_dir/pipeline.log"
    ;;
  prepare_rag)
    "${spark_submit[@]}" "$code_root/ecommerce_rag/rag/prepare.py" --environment aws \
      --summary-output "$report_dir/preparation_summary.json" \
      2>&1 | tee "$report_dir/prepare_rag.log"
    ;;
  manifest)
    "${spark_submit[@]}" "$code_root/ecommerce_rag/cloud/validation.py" manifest \
      --environment aws --output "$report_dir/cloud_manifest.json" \
      2>&1 | tee "$report_dir/manifest.log"
    ;;
  scalability)
    "${spark_submit[@]}" \
      --conf spark.executor.cores=4 \
      --conf spark.executor.memory=10g \
      --conf spark.sql.shuffle.partitions=32 \
      "$code_root/ecommerce_rag/scalability/phase9.py" --environment aws --mode all \
      --output-root "$report_dir" --factors "$@" --repetitions 3 \
      --warmup-runs 1 --force 2>&1 | tee "$report_dir/scalability.log"
    ;;
  *)
    echo "Unsupported Phase 10 action: $action" >&2
    exit 2
    ;;
esac

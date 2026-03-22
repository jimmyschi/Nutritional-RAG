from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def _load_questions(eval_set_path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with eval_set_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = line.strip()
            if not payload:
                continue
            questions.append(json.loads(payload))
    if not questions:
        raise ValueError(f"No questions found in {eval_set_path}")
    return questions


def _parse_metrics(metrics_text: str) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for line in metrics_text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        try:
            parsed[name] = float(value)
        except ValueError:
            continue
    return parsed


def _fetch_metrics(metrics_url: str) -> dict[str, float]:
    response = requests.get(metrics_url, timeout=15)
    response.raise_for_status()
    return _parse_metrics(response.text)


def _metric_delta(before: dict[str, float], after: dict[str, float], prefix: str) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for key, after_value in after.items():
        if not key.startswith(prefix):
            continue
        deltas[key] = after_value - before.get(key, 0.0)
    return deltas


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _build_payload(question_row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "question": str(question_row["question"]),
        "top_k": args.top_k or int(question_row.get("top_k", 5)),
        "use_cache": args.use_cache,
        "generate_answer": args.generate_answer,
    }
    if args.rerank_candidate_multiplier is not None:
        payload["rerank_candidate_multiplier"] = args.rerank_candidate_multiplier
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send repeat traffic to the API and summarize Prometheus metric deltas"
    )
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8001/metrics")
    parser.add_argument(
        "--eval-set",
        default="data/eval/nutrition_eval_set.ndjson",
        help="NDJSON file containing questions to replay",
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--rerank-candidate-multiplier", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument(
        "--generate-answer",
        action="store_true",
        help="Generate final answers instead of retrieval-only traffic",
    )
    parser.add_argument("--output", default=None, help="Optional path for JSON summary output")
    args = parser.parse_args()

    questions = _load_questions(Path(args.eval_set))
    before_metrics = _fetch_metrics(args.metrics_url)

    request_results: list[dict[str, Any]] = []
    for round_index in range(args.rounds):
        for question_row in questions:
            payload = _build_payload(question_row, args)
            started_at = time.perf_counter()
            try:
                response = requests.post(
                    f"{args.api_base_url.rstrip('/')}/query",
                    json=payload,
                    timeout=120,
                )
                latency_ms = (time.perf_counter() - started_at) * 1000
                response.raise_for_status()
                body = response.json()
                request_results.append(
                    {
                        "round": round_index + 1,
                        "question": payload["question"],
                        "status_code": response.status_code,
                        "latency_ms": latency_ms,
                        "cache_hit": bool(body.get("cache_hit", False)),
                        "citation_count": len(body.get("citations", []) or []),
                    }
                )
            except requests.RequestException as error:
                request_results.append(
                    {
                        "round": round_index + 1,
                        "question": payload["question"],
                        "status_code": 0,
                        "latency_ms": (time.perf_counter() - started_at) * 1000,
                        "cache_hit": False,
                        "citation_count": 0,
                        "error": str(error),
                    }
                )
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)

    after_metrics = _fetch_metrics(args.metrics_url)
    durations = [row["latency_ms"] for row in request_results if row["status_code"] == 200]
    cache_hits = [1.0 for row in request_results if row.get("cache_hit")]
    errors = [row for row in request_results if row["status_code"] != 200]
    prometheus_requests = _metric_delta(
        before_metrics, after_metrics, "nutritional_rag_query_requests_total"
    )
    prometheus_cache = _metric_delta(
        before_metrics, after_metrics, "nutritional_rag_query_cache_checks_total"
    )

    cache_hit_delta = sum(
        value for key, value in prometheus_cache.items() if 'result="hit"' in key
    )
    cache_miss_delta = sum(
        value for key, value in prometheus_cache.items() if 'result="miss"' in key
    )
    duration_sum = after_metrics.get("nutritional_rag_query_duration_seconds_sum", 0.0) - before_metrics.get(
        "nutritional_rag_query_duration_seconds_sum", 0.0
    )
    duration_count = after_metrics.get(
        "nutritional_rag_query_duration_seconds_count", 0.0
    ) - before_metrics.get("nutritional_rag_query_duration_seconds_count", 0.0)
    citation_score_sum = after_metrics.get(
        "nutritional_rag_query_mean_citation_score_sum", 0.0
    ) - before_metrics.get("nutritional_rag_query_mean_citation_score_sum", 0.0)
    citation_score_count = after_metrics.get(
        "nutritional_rag_query_mean_citation_score_count", 0.0
    ) - before_metrics.get("nutritional_rag_query_mean_citation_score_count", 0.0)

    summary = {
        "request_count": len(request_results),
        "successful_requests": len(request_results) - len(errors),
        "error_rate": _safe_divide(len(errors), len(request_results)),
        "client_latency_ms_mean": statistics.mean(durations) if durations else 0.0,
        "client_latency_ms_p95": (
            sorted(durations)[int(0.95 * (len(durations) - 1))] if durations else 0.0
        ),
        "client_cache_hit_rate": _safe_divide(sum(cache_hits), len(request_results)),
        "prometheus_cache_hit_rate": _safe_divide(cache_hit_delta, cache_hit_delta + cache_miss_delta),
        "prometheus_latency_seconds_mean": _safe_divide(duration_sum, duration_count),
        "prometheus_mean_citation_score": _safe_divide(citation_score_sum, citation_score_count),
        "prometheus_request_deltas": prometheus_requests,
        "prometheus_cache_deltas": prometheus_cache,
        "errors": errors,
    }

    print(json.dumps(summary, indent=2))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
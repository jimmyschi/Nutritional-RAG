from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def _load_eval_set(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _keyword_hit_rate(answer: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 0.0
    answer_lower = answer.lower()
    hits = sum(1 for keyword in expected_keywords if keyword.lower() in answer_lower)
    return hits / len(expected_keywords)


def _source_recall(citations: list[dict[str, Any]], expected_source_ids: list[str]) -> float:
    if not expected_source_ids:
        return 0.0
    actual = {str(citation.get("source_id", "")).lower() for citation in citations}
    expected = {source_id.lower() for source_id in expected_source_ids}
    if not expected:
        return 0.0
    return len(actual & expected) / len(expected)


def _mean_or_zero(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _evaluate_one(
    api_base_url: str,
    row: dict[str, Any],
    timeout_seconds: int,
    *,
    rerank_candidate_multiplier: int | None = None,
    use_cache: bool = False,
    generate_answer: bool = True,
) -> dict[str, Any]:
    question = str(row.get("question", "")).strip()
    top_k = int(row.get("top_k", 5))
    expected_keywords = [str(item) for item in row.get("expected_keywords", [])]
    expected_sources = [str(item) for item in row.get("expected_source_ids", [])]

    payload = {
        "question": question,
        "top_k": top_k,
        "use_cache": use_cache,
        "generate_answer": generate_answer,
    }
    if rerank_candidate_multiplier is not None:
        payload["rerank_candidate_multiplier"] = rerank_candidate_multiplier

    started_at = time.perf_counter()
    try:
        response = requests.post(
            f"{api_base_url.rstrip('/')}/query",
            json=payload,
            timeout=timeout_seconds,
        )
        latency_ms = (time.perf_counter() - started_at) * 1000
    except requests.RequestException as error:
        return {
            "question": question,
            "status_code": 0,
            "error": str(error),
            "latency_ms": (time.perf_counter() - started_at) * 1000,
            "keyword_hit_rate": 0.0,
            "source_recall": 0.0,
            "mean_citation_score": 0.0,
            "cache_hit": 0.0,
            "citation_count": 0.0,
        }

    if response.status_code != 200:
        return {
            "question": question,
            "status_code": response.status_code,
            "error": response.text[:500],
            "latency_ms": latency_ms,
            "keyword_hit_rate": 0.0,
            "source_recall": 0.0,
            "mean_citation_score": 0.0,
            "cache_hit": 0.0,
            "citation_count": 0.0,
        }

    payload = response.json()
    answer = str(payload.get("answer", ""))
    citations = payload.get("citations", []) or []
    scores = [float(citation.get("score", 0.0)) for citation in citations]

    return {
        "question": question,
        "status_code": response.status_code,
        "error": "",
        "latency_ms": latency_ms,
        "keyword_hit_rate": _keyword_hit_rate(answer, expected_keywords),
        "source_recall": _source_recall(citations, expected_sources),
        "mean_citation_score": _mean_or_zero(scores),
        "cache_hit": 1.0 if bool(payload.get("cache_hit", False)) else 0.0,
        "citation_count": float(len(citations)),
    }


def _log_to_mlflow(
    *,
    tracking_uri: str,
    experiment_name: str,
    eval_set_path: Path,
    api_base_url: str,
    top_k_override: int | None,
    rerank_candidate_multiplier: int | None,
    use_cache: bool,
    generate_answer: bool,
    results: list[dict[str, Any]],
    run_name: str = "api-eval",
) -> None:
    try:
        import mlflow
    except ModuleNotFoundError:
        print("MLflow not installed in this environment. Install with: pip install -e '.[ml]'")
        return

    latencies = [float(row["latency_ms"]) for row in results]
    keyword_hits = [float(row["keyword_hit_rate"]) for row in results]
    source_recalls = [float(row["source_recall"]) for row in results]
    mean_scores = [float(row["mean_citation_score"]) for row in results]
    cache_hits = [float(row["cache_hit"]) for row in results]
    citation_counts = [float(row["citation_count"]) for row in results]
    errors = [row for row in results if int(row["status_code"]) != 200]
    p95_latency = (
        sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "eval_set_path": str(eval_set_path),
                "api_base_url": api_base_url,
                "question_count": len(results),
                "top_k_override": top_k_override if top_k_override is not None else "none",
                "rerank_candidate_multiplier": (
                    rerank_candidate_multiplier
                    if rerank_candidate_multiplier is not None
                    else "none"
                ),
                "use_cache": use_cache,
                "generate_answer": generate_answer,
            }
        )
        mlflow.log_metrics(
            {
                "eval_error_rate": len(errors) / max(1, len(results)),
                "eval_latency_ms_mean": _mean_or_zero(latencies),
                "eval_latency_ms_p95": p95_latency,
                "eval_keyword_hit_rate_mean": _mean_or_zero(keyword_hits),
                "eval_source_recall_mean": _mean_or_zero(source_recalls),
                "eval_mean_citation_score": _mean_or_zero(mean_scores),
                "eval_cache_hit_rate": _mean_or_zero(cache_hits),
                "eval_citation_count_mean": _mean_or_zero(citation_counts),
            }
        )
        mlflow.log_text(json.dumps(results, indent=2), "eval_results.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a lightweight RAG API evaluation and log to MLflow"
    )
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--eval-set",
        default="data/eval/nutrition_eval_set.ndjson",
        help="Path to NDJSON eval set",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=None, help="Override top_k from eval rows")
    parser.add_argument(
        "--rerank-candidate-multiplier",
        type=int,
        default=None,
        help="Override rerank candidate multiplier for this evaluation run",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Allow cached API responses during evaluation",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip answer generation and evaluate retrieval only",
    )
    parser.add_argument("--mlflow-tracking-uri", default="http://localhost:5001")
    parser.add_argument("--mlflow-experiment", default="nutritional-rag-eval")
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    eval_set_path = Path(args.eval_set)
    rows = _load_eval_set(eval_set_path)
    if not rows:
        raise SystemExit(f"No eval rows found in {eval_set_path}")

    results: list[dict[str, Any]] = []
    for row in rows:
        if args.top_k is not None:
            row = dict(row)
            row["top_k"] = args.top_k
        result = _evaluate_one(
            args.api_base_url,
            row,
            args.timeout_seconds,
            rerank_candidate_multiplier=args.rerank_candidate_multiplier,
            use_cache=args.use_cache,
            generate_answer=not args.skip_generation,
        )
        results.append(result)

    latencies = [float(row["latency_ms"]) for row in results]
    keyword_hit_values = [float(row["keyword_hit_rate"]) for row in results]
    source_recall_values = [float(row["source_recall"]) for row in results]
    citation_score_values = [float(row["mean_citation_score"]) for row in results]
    errors = [row for row in results if int(row["status_code"]) != 200]
    print(f"Evaluated {len(results)} questions against {args.api_base_url}")
    print(f"Error rate: {len(errors) / max(1, len(results)):.2%}")
    print(f"Mean latency (ms): {_mean_or_zero(latencies):.2f}")
    print(f"Mean keyword hit rate: {_mean_or_zero(keyword_hit_values):.4f}")
    print(f"Mean source recall: {_mean_or_zero(source_recall_values):.4f}")
    print(f"Mean citation score: {_mean_or_zero(citation_score_values):.4f}")

    if not args.skip_mlflow:
        _log_to_mlflow(
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            eval_set_path=eval_set_path,
            api_base_url=args.api_base_url,
            top_k_override=args.top_k,
            rerank_candidate_multiplier=args.rerank_candidate_multiplier,
            use_cache=args.use_cache,
            generate_answer=not args.skip_generation,
            results=results,
        )


if __name__ == "__main__":
    main()

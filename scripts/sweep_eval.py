from __future__ import annotations

import argparse
from pathlib import Path

from evaluate_rag import _evaluate_one, _load_eval_set, _log_to_mlflow, _mean_or_zero


def _parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of integers")
    return [int(item) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a parameter sweep over top_k and rerank multiplier values"
    )
    parser.add_argument("--api-base-url", default="http://127.0.0.1:8001")
    parser.add_argument(
        "--eval-set",
        default="data/eval/nutrition_eval_set.ndjson",
        help="Path to NDJSON eval set",
    )
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--top-k-values", type=_parse_int_list, default=[3, 5, 8])
    parser.add_argument(
        "--rerank-candidate-multipliers",
        type=_parse_int_list,
        default=[1, 2, 3],
    )
    parser.add_argument("--use-cache", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default="http://127.0.0.1:5001")
    parser.add_argument("--mlflow-experiment", default="nutritional-rag-sweep")
    parser.add_argument("--skip-mlflow", action="store_true")
    args = parser.parse_args()

    eval_set_path = Path(args.eval_set)
    rows = _load_eval_set(eval_set_path)
    if not rows:
        raise SystemExit(f"No eval rows found in {eval_set_path}")

    for top_k in args.top_k_values:
        for rerank_multiplier in args.rerank_candidate_multipliers:
            results = []
            for row in rows:
                current_row = dict(row)
                current_row["top_k"] = top_k
                result = _evaluate_one(
                    args.api_base_url,
                    current_row,
                    args.timeout_seconds,
                    rerank_candidate_multiplier=rerank_multiplier,
                    use_cache=args.use_cache,
                    generate_answer=False,
                )
                results.append(result)

            error_rate = len([row for row in results if int(row["status_code"]) != 200]) / max(
                1, len(results)
            )
            latency_mean = _mean_or_zero([float(row["latency_ms"]) for row in results])
            keyword_mean = _mean_or_zero([float(row["keyword_hit_rate"]) for row in results])
            source_mean = _mean_or_zero([float(row["source_recall"]) for row in results])
            score_mean = _mean_or_zero([float(row["mean_citation_score"]) for row in results])

            print(
                " | ".join(
                    [
                        f"top_k={top_k}",
                        f"rerank_multiplier={rerank_multiplier}",
                        f"error_rate={error_rate:.2%}",
                        f"latency_ms_mean={latency_mean:.2f}",
                        f"keyword_hit_rate_mean={keyword_mean:.4f}",
                        f"source_recall_mean={source_mean:.4f}",
                        f"citation_score_mean={score_mean:.4f}",
                    ]
                )
            )

            if args.skip_mlflow:
                continue

            _log_to_mlflow(
                tracking_uri=args.mlflow_tracking_uri,
                experiment_name=args.mlflow_experiment,
                eval_set_path=eval_set_path,
                api_base_url=args.api_base_url,
                top_k_override=top_k,
                rerank_candidate_multiplier=rerank_multiplier,
                use_cache=args.use_cache,
                generate_answer=False,
                results=results,
                run_name=f"sweep-topk-{top_k}-rerank-{rerank_multiplier}",
            )


if __name__ == "__main__":
    main()

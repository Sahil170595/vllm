# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reduce ``vllm bench sweep serve`` results to a concurrency-boundary table.

This subcommand post-processes the per-run results produced by
``vllm bench sweep serve`` (or ``serve_workload``) and reports, for each
workload cell, how tail latency amplifies as client concurrency grows
relative to the lowest-concurrency baseline (``N=1`` by default).

The headline metric is the p95 end-to-end latency multiplier versus the
baseline concurrency, computed per repetition and then averaged across
repetitions. Unlike absolute throughput or latency, this ratio is normalized
within each cell, so it is more comparable across hardware than raw tok/s.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_GROUP_BY = ["_benchmark_name"]
DEFAULT_METRICS = [
    "output_throughput",
    "p50_e2el_ms",
    "p95_e2el_ms",
    "p50_ttft_ms",
    "p95_ttft_ms",
]
DEFAULT_MULTIPLIER_METRICS = ["p95_e2el_ms", "p95_ttft_ms"]


def compute_boundary_table(
    records: list[dict[str, object]],
    *,
    group_by: list[str] | None = None,
    concurrency_var: str = "max_concurrency",
    run_var: str = "run_number",
    baseline: float = 1,
    metrics: list[str] | None = None,
    multiplier_metrics: list[str] | None = None,
) -> "pd.DataFrame":
    """Compute a concurrency-boundary table from sweep run records.

    For each workload cell (every column in ``group_by``) the latency at each
    concurrency level is divided by the same cell's baseline-concurrency
    latency, per repetition, and the resulting multipliers are then averaged
    across repetitions. Raw metrics are averaged across repetitions as-is.

    Args:
        records: Per-run result dicts, e.g. the concatenation of every
            ``summary.json`` emitted by ``vllm bench sweep serve``.
        group_by: Columns identifying a workload cell (everything except
            concurrency and repetition). Defaults to ``["_benchmark_name"]``.
        concurrency_var: Column holding the client concurrency. Defaults to
            ``"max_concurrency"``.
        run_var: Column holding the repetition index. Defaults to
            ``"run_number"``.
        baseline: Concurrency value used as the per-cell denominator for
            latency multipliers. Defaults to ``1``.
        metrics: Raw metric columns to average per cell. Columns absent from
            the records are skipped.
        multiplier_metrics: Metric columns for which a vs-baseline multiplier
            is computed (per repetition, then averaged).

    Returns:
        A ``pandas.DataFrame`` with one row per ``(cell, concurrency)`` holding
        the averaged raw metrics, the averaged vs-baseline multipliers, the
        repetition count ``n_runs``, and (when present) averaged
        ``completed`` / ``failed`` request counts.

    Raises:
        ValueError: If ``records`` is empty, a grouping/concurrency/run column
            is missing, or none of ``multiplier_metrics`` are present. The
            latter most often means ``--metric-percentiles 50,95,99`` was not
            passed inside ``--bench-cmd``, so only ``p99_*`` columns exist.
    """
    import pandas as pd

    if not records:
        raise ValueError("No sweep records to reduce.")

    group_by = list(group_by) if group_by else list(DEFAULT_GROUP_BY)
    metrics = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    multiplier_metrics = (
        list(multiplier_metrics)
        if multiplier_metrics is not None
        else list(DEFAULT_MULTIPLIER_METRICS)
    )

    df = pd.DataFrame.from_records(records)

    required = [*group_by, concurrency_var, run_var]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Sweep records are missing required column(s): {missing}. "
            f"Available columns: {sorted(df.columns)}"
        )

    present_multiplier = [m for m in multiplier_metrics if m in df.columns]
    if not present_multiplier:
        raise ValueError(
            f"None of the multiplier metrics {multiplier_metrics} are present "
            "in the sweep records. If you expected p95 columns, pass "
            "'--metric-percentiles 50,95,99' inside --bench-cmd: the sweep "
            "runner only sets --percentile-metrics, so the default leaves only "
            f"p99_* columns. Available columns: {sorted(df.columns)}"
        )

    present_metrics = [m for m in metrics if m in df.columns]

    # Per-(cell, repetition) baseline value for each multiplier metric.
    key_cols = [*group_by, run_var]
    base = (
        df[df[concurrency_var] == baseline]
        .groupby(key_cols, dropna=False)[present_multiplier]
        .mean()
        .add_prefix("_baseline_")
        .reset_index()
    )
    merged = df.merge(base, on=key_cols, how="left")

    mult_cols = []
    for m in present_multiplier:
        col = f"{m}_multiplier_vs_baseline"
        merged[col] = merged[m] / merged[f"_baseline_{m}"]
        mult_cols.append(col)

    agg_map: dict[str, str] = {m: "mean" for m in present_metrics}
    for col in mult_cols:
        agg_map[col] = "mean"
    for count_col in ("completed", "failed"):
        if count_col in merged.columns:
            agg_map[count_col] = "mean"

    grouped = merged.groupby([*group_by, concurrency_var], dropna=False)
    table = grouped.agg(agg_map)
    table["n_runs"] = grouped.size()
    table = table.reset_index().sort_values([*group_by, concurrency_var])
    return table.reset_index(drop=True)


def _load_records(output_dir: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in output_dir.rglob("**/summary.json"):
        with path.open("rb") as f:
            records.extend(json.load(f))
    return records


@dataclass
class SweepBoundaryArgs:
    output_dir: Path
    output_file: Path
    group_by: list[str]
    concurrency_var: str
    run_var: str
    baseline: float
    metrics: list[str]
    multiplier_metrics: list[str]

    parser_name: ClassVar[str] = "boundary"
    parser_help: ClassVar[str] = (
        "Reduce sweep results to a concurrency-boundary table "
        "(p95 latency amplification vs the baseline concurrency)."
    )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        output_dir = Path(args.EXPERIMENT_DIR)
        if not output_dir.exists():
            raise ValueError(f"No parameter sweep results under {output_dir}")

        return cls(
            output_dir=output_dir,
            output_file=output_dir / args.output_file,
            group_by=args.group_by.split(","),
            concurrency_var=args.concurrency_var,
            run_var=args.run_var,
            baseline=args.baseline,
            metrics=args.metrics.split(","),
            multiplier_metrics=args.multiplier_metrics.split(","),
        )

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "EXPERIMENT_DIR",
            type=str,
            help="The directory containing the sweep results to reduce.",
        )
        parser.add_argument(
            "--output-file",
            type=str,
            default="boundary.csv",
            help="Where to save the boundary table, relative to EXPERIMENT_DIR. "
            "Default: 'boundary.csv'.",
        )
        parser.add_argument(
            "--group-by",
            type=str,
            default=",".join(DEFAULT_GROUP_BY),
            help="A comma-separated list of columns identifying a workload cell "
            "(everything except concurrency and repetition). "
            f"Default: '{','.join(DEFAULT_GROUP_BY)}'.",
        )
        parser.add_argument(
            "--concurrency-var",
            type=str,
            default="max_concurrency",
            help="The column holding the client concurrency. "
            "Default: 'max_concurrency'.",
        )
        parser.add_argument(
            "--run-var",
            type=str,
            default="run_number",
            help="The column holding the repetition index. Default: 'run_number'.",
        )
        parser.add_argument(
            "--baseline",
            type=float,
            default=1,
            help="The concurrency value used as the per-cell denominator for "
            "latency multipliers. Default: 1.",
        )
        parser.add_argument(
            "--metrics",
            type=str,
            default=",".join(DEFAULT_METRICS),
            help="A comma-separated list of raw metric columns to average per "
            f"cell. Default: '{','.join(DEFAULT_METRICS)}'.",
        )
        parser.add_argument(
            "--multiplier-metrics",
            type=str,
            default=",".join(DEFAULT_MULTIPLIER_METRICS),
            help="A comma-separated list of metric columns for which a "
            "vs-baseline multiplier is computed. "
            f"Default: '{','.join(DEFAULT_MULTIPLIER_METRICS)}'.",
        )

        return parser


def run_main(args: SweepBoundaryArgs):
    records = _load_records(args.output_dir)
    if not records:
        raise ValueError(
            f"Did not find any parameter sweep results under {args.output_dir}"
        )

    table = compute_boundary_table(
        records,
        group_by=args.group_by,
        concurrency_var=args.concurrency_var,
        run_var=args.run_var,
        baseline=args.baseline,
        metrics=args.metrics,
        multiplier_metrics=args.multiplier_metrics,
    )

    table.to_csv(args.output_file, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved boundary table to {args.output_file}")

    return table


def main(args: argparse.Namespace):
    run_main(SweepBoundaryArgs.from_cli_args(args))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=SweepBoundaryArgs.parser_help)
    SweepBoundaryArgs.add_cli_args(parser)

    main(parser.parse_args())

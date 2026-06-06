# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math

import pytest

from vllm.benchmarks.sweep.boundary import compute_boundary_table


def _records():
    """Two cells x {N=1, N=32} x 3 reps with known p95 latencies.

    ``long_prefill_8k`` is built so the per-rep p95 E2EL multipliers at N=32
    are exactly 3.42 / 4.41 / 4.45 (mean 4.0933), reproducing the verified
    TR164 fixture. ``long_decode`` stays near 1.0x.
    """
    rows: list[dict[str, object]] = []

    n32_p95 = {0: 3420.0, 1: 4410.0, 2: 4450.0}
    for rep in range(3):
        rows.append(
            {
                "_benchmark_name": "long_prefill_8k",
                "max_concurrency": 1,
                "run_number": rep,
                "p95_e2el_ms": 1000.0,
                "p50_e2el_ms": 900.0,
                "p95_ttft_ms": 500.0,
                "p50_ttft_ms": 450.0,
                "output_throughput": 100.0 + rep,
                "completed": 10,
                "failed": 0,
            }
        )
        rows.append(
            {
                "_benchmark_name": "long_prefill_8k",
                "max_concurrency": 32,
                "run_number": rep,
                "p95_e2el_ms": n32_p95[rep],
                "p50_e2el_ms": 3000.0,
                "p95_ttft_ms": 2000.0,
                "p50_ttft_ms": 1800.0,
                "output_throughput": 640.0 + rep,
                "completed": 320,
                "failed": 0,
            }
        )

    for rep in range(3):
        rows.append(
            {
                "_benchmark_name": "long_decode",
                "max_concurrency": 1,
                "run_number": rep,
                "p95_e2el_ms": 2000.0,
                "p50_e2el_ms": 1900.0,
                "p95_ttft_ms": 100.0,
                "p50_ttft_ms": 90.0,
                "output_throughput": 200.0,
                "completed": 10,
                "failed": 0,
            }
        )
        rows.append(
            {
                "_benchmark_name": "long_decode",
                "max_concurrency": 32,
                "run_number": rep,
                "p95_e2el_ms": 2600.0,
                "p50_e2el_ms": 2500.0,
                "p95_ttft_ms": 130.0,
                "p50_ttft_ms": 120.0,
                "output_throughput": 6000.0,
                "completed": 320,
                "failed": 0,
            }
        )

    return rows


def _cell(table, workload, concurrency):
    df = table[
        (table["_benchmark_name"] == workload)
        & (table["max_concurrency"] == concurrency)
    ]
    assert len(df) == 1
    return df.iloc[0]


def test_per_rep_then_mean_multiplier():
    """The multiplier is the mean of per-rep ratios, not ratio of means."""
    table = compute_boundary_table(_records())
    row = _cell(table, "long_prefill_8k", 32)
    assert row["p95_e2el_ms_multiplier_vs_baseline"] == pytest.approx(4.0933, abs=1e-3)


def test_baseline_multiplier_is_one():
    table = compute_boundary_table(_records())
    for workload in ("long_prefill_8k", "long_decode"):
        row = _cell(table, workload, 1)
        assert row["p95_e2el_ms_multiplier_vs_baseline"] == pytest.approx(1.0)


def test_raw_metrics_are_averaged_across_reps():
    table = compute_boundary_table(_records())
    row = _cell(table, "long_prefill_8k", 1)
    assert row["output_throughput"] == pytest.approx(101.0)  # mean(100, 101, 102)
    assert row["n_runs"] == 3
    assert row["completed"] == pytest.approx(10.0)


def test_ttft_multiplier_computed():
    table = compute_boundary_table(_records())
    row = _cell(table, "long_prefill_8k", 32)
    # 2000 / 500 = 4.0 for every rep.
    assert row["p95_ttft_ms_multiplier_vs_baseline"] == pytest.approx(4.0)


def test_missing_baseline_yields_nan_not_crash():
    # A cell whose only rows are at N=32 (no baseline) must not crash.
    records = _records() + [
        {
            "_benchmark_name": "orphan",
            "max_concurrency": 32,
            "run_number": 0,
            "p95_e2el_ms": 9000.0,
            "p95_ttft_ms": 9000.0,
            "output_throughput": 1.0,
        }
    ]
    table = compute_boundary_table(records)
    row = _cell(table, "orphan", 32)
    assert math.isnan(row["p95_e2el_ms_multiplier_vs_baseline"])


def test_missing_p95_column_raises_actionable_error():
    # Only p99 present (the sweep default) -> the reducer must explain the fix.
    records = [
        {
            "_benchmark_name": "w",
            "max_concurrency": 1,
            "run_number": 0,
            "p99_e2el_ms": 100.0,
        },
        {
            "_benchmark_name": "w",
            "max_concurrency": 32,
            "run_number": 0,
            "p99_e2el_ms": 400.0,
        },
    ]
    with pytest.raises(ValueError, match="--metric-percentiles"):
        compute_boundary_table(records)


def test_empty_records_raises():
    with pytest.raises(ValueError, match="No sweep records"):
        compute_boundary_table([])


def test_missing_required_column_raises():
    records = [{"_benchmark_name": "w", "max_concurrency": 1, "p95_e2el_ms": 1.0}]
    with pytest.raises(ValueError, match="run_number"):
        compute_boundary_table(records)

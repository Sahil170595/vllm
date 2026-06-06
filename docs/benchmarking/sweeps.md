# Parameter Sweeps

`vllm bench sweep` is a suite of commands designed to run benchmarks across multiple configurations and compare them by visualizing the results.

## Online Benchmark

### Basic

`vllm bench sweep serve` starts `vllm serve` and iteratively runs `vllm bench serve` for each server configuration.

!!! tip
    If you only need to run benchmarks for a single server configuration, consider using [GuideLLM](https://github.com/vllm-project/guidellm), an established performance benchmarking framework with live progress updates and automatic report generation. It is also more flexible than `vllm bench serve` in terms of dataset loading, request formatting, and workload patterns.

Follow these steps to run the script:

1. Construct the base command to `vllm serve`, and pass it to the `--serve-cmd` option.
2. Construct the base command to `vllm bench serve`, and pass it to the `--bench-cmd` option.
3. (Optional) If you would like to vary the settings of `vllm serve`, create a new JSON file and populate it with the parameter combinations you want to test. Pass the file path to `--serve-params`.

    - Example: Tuning `--max-num-seqs` and `--max-num-batched-tokens`:

    ```json
    [
        {
            "max_num_seqs": 32,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 1024
        },
        {
            "max_num_seqs": 64,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 2048
        },
        {
            "max_num_seqs": 128,
            "max_num_batched_tokens": 4096
        },
        {
            "max_num_seqs": 256,
            "max_num_batched_tokens": 4096
        }
    ]
    ```

4. (Optional) If you would like to vary the settings of `vllm bench serve`, create a new JSON file and populate it with the parameter combinations you want to test. Pass the file path to `--bench-params`.

    - Example: Using different input/output lengths for random dataset:

    ```json
    [
        {
            "_benchmark_name": "scenario_A",
            "random_input_len": 128,
            "random_output_len": 32
        },
        {
            "_benchmark_name": "scenario_B",
            "random_input_len": 256,
            "random_output_len": 64
        },
        {
            "_benchmark_name": "scenario_C",
            "random_input_len": 512,
            "random_output_len": 128
        }
    ]
    ```

5. Set `--output-dir` and optionally `--experiment-name` to control where to save the results.

Example command:

```bash
vllm bench sweep serve \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json' \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```

By default, each parameter combination is benchmarked 3 times to make the results more reliable. You can adjust the number of runs by setting `--num-runs`.

!!! important
    If both `--serve-params` and `--bench-params` are passed, the script will iterate over the Cartesian product between them.
    You can use `--dry-run` to preview the commands to be run.

    We only start the server once for each `--serve-params`, and keep it running for multiple `--bench-params`.
    Between each benchmark run, we call all `/reset_*_cache` endpoints to get a clean slate for the next run.
    In case you are using a custom `--serve-cmd`, you can override the commands used for resetting the state by setting `--after-bench-cmd`.

!!! note
    You should set `_benchmark_name` to provide a human-readable name for parameter combinations involving many variables.
    This becomes mandatory if the file name would otherwise exceed the maximum path length allowed by the filesystem.

!!! tip
    You can use the `--resume` option to continue the parameter sweep if an unexpected error occurs, e.g., timeout when connecting to HF Hub.

### Workload Explorer

`vllm bench sweep serve_workload` is a variant of `vllm bench sweep serve` that explores different workload levels in order to find the tradeoff between latency and throughput. The results can also be [visualized](#visualization) to determine the feasible SLAs.

The workload can be expressed in terms of request rate or concurrency (choose using `--workload-var`).

Example command:

```bash
vllm bench sweep serve_workload \
    --serve-cmd 'vllm serve meta-llama/Llama-2-7b-chat-hf' \
    --bench-cmd 'vllm bench serve --model meta-llama/Llama-2-7b-chat-hf --backend vllm --endpoint /v1/completions --dataset-name sharegpt --dataset-path benchmarks/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 100' \
    --workload-var max_concurrency \
    --serve-params benchmarks/serve_hparams.json \
    --bench-params benchmarks/bench_hparams.json \
    --num-runs 1 \
    --output-dir benchmarks/results \
    --experiment-name demo
```

The algorithm for exploring different workload levels can be summarized as follows:

1. Run the benchmark by sending requests one at a time (serial inference, lowest workload). This results in the lowest possible latency and throughput.
2. Run the benchmark by sending all requests at once (batch inference, highest workload). This results in the highest possible latency and throughput.
3. Estimate the value of `workload_var` corresponding to Step 2.
4. Run the benchmark over intermediate values of `workload_var` uniformly using the remaining iterations.

You can override the number of iterations in the algorithm by setting `--workload-iters`.

!!! tip
    This is our equivalent of [GuideLLM's `--profile sweep`](https://github.com/vllm-project/guidellm/blob/v0.5.3/src/guidellm/benchmark/profiles.py#L575).

    In general, `--workload-var max_concurrency` produces more reliable results because it directly controls the workload imposed on the vLLM engine.
    Nevertheless, we default to `--workload-var request_rate` to maintain similar behavior as GuideLLM.

## Startup Benchmark

`vllm bench sweep startup` runs `vllm bench startup` across parameter combinations to compare cold/warm startup time for different engine settings.

Follow these steps to run the script:

1. (Optional) Construct the base command to `vllm bench startup`, and pass it to `--startup-cmd` (default: `vllm bench startup`).
2. (Optional) Reuse a `--serve-params` JSON from `vllm bench sweep serve` to vary engine settings. Only parameters supported by `vllm bench startup` are applied.
3. (Optional) Create a `--startup-params` JSON to vary startup-specific options like iteration counts.
4. Determine where you want to save the results, and pass that to `--output-dir`.

Example `--serve-params`:

```json
[
    {
        "_benchmark_name": "tp1",
        "model": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": 0.9
    },
    {
        "_benchmark_name": "tp2",
        "model": "Qwen/Qwen3-0.6B",
        "tensor_parallel_size": 2,
        "gpu_memory_utilization": 0.9
    }
]
```

Example `--startup-params`:

```json
[
    {
        "_benchmark_name": "qwen3-0.6",
        "num_iters_cold": 2,
        "num_iters_warmup": 1,
        "num_iters_warm": 2
    }
]
```

Example command:

```bash
vllm bench sweep startup \
    --startup-cmd 'vllm bench startup --model Qwen/Qwen3-0.6B' \
    --serve-params benchmarks/serve_hparams.json \
    --startup-params benchmarks/startup_hparams.json \
    --output-dir benchmarks/results \
    --experiment-name demo
```

!!! important
    By default, unsupported parameters in `--serve-params` or `--startup-params` are ignored with a warning.
    Use `--strict-params` to fail fast on unknown keys.

## Visualization

### Basic

`vllm bench sweep plot` can be used to plot performance curves from parameter sweep results.

Control the variables to plot via `--var-x` and `--var-y`, optionally applying `--filter-by` and `--bin-by` to the values. The plot is organized according to `--fig-by`, `--row-by`, `--col-by`, and `--curve-by`.

Example commands for visualizing [Workload Explorer](#workload-explorer) results:

```bash
EXPERIMENT_DIR=${1:-"benchmarks/results/demo"}

# Latency increases as the workload increases
vllm bench sweep plot $EXPERIMENT_DIR \
    --var-x max_concurrency \
    --var-y median_ttft_ms \
    --col-by _benchmark_name \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --fig-name latency_curve

# Throughput saturates as workload increases
vllm bench sweep plot $EXPERIMENT_DIR \
    --var-x max_concurrency \
    --var-y total_token_throughput \
    --col-by _benchmark_name \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --fig-name throughput_curve

# Tradeoff between latency and throughput
vllm bench sweep plot $EXPERIMENT_DIR \
    --var-x total_token_throughput \
    --var-y median_ttft_ms \
    --col-by _benchmark_name \
    --curve-by max_num_seqs,max_num_batched_tokens \
    --fig-name latency_throughput
```

!!! tip
    You can use `--dry-run` to preview the figures to be plotted.

### Pareto chart

`vllm bench sweep plot_pareto` helps pick configurations that balance per-user and per-GPU throughput.

Higher concurrency or batch size can raise GPU efficiency (per-GPU), but can add per user latency; lower concurrency improves per-user rate but underutilizes GPUs; The Pareto frontier shows the best achievable pairs across your runs.

- x-axis: tokens/s/user = `output_throughput` ÷ concurrency (`--user-count-var`, default `max_concurrency`, fallback `max_concurrent_requests`).
- y-axis: tokens/s/GPU = `output_throughput` ÷ GPU count (`--gpu-count-var` if set; else gpu_count is TP×PP*DP).
- Output: a single figure at `OUTPUT_DIR/pareto/PARETO.png`.
- Show the configuration used in each data point `--label-by` (default: `max_concurrency,gpu_count`).

Example:

```bash
EXPERIMENT_DIR=${1:-"benchmarks/results/demo"}

vllm bench sweep plot_pareto $EXPERIMENT_DIR \
  --label-by max_concurrency,tensor_parallel_size,pipeline_parallel_size
```

!!! tip
    You can use `--dry-run` to preview the figures to be plotted.

### Boundary table

`vllm bench sweep boundary` reduces sweep results into a *concurrency-boundary table*: for each workload cell it reports how tail latency amplifies as client concurrency grows relative to the lowest-concurrency baseline (`N=1` by default). The headline metric, `p95_e2el_ms_multiplier_vs_baseline`, is computed per repetition and then averaged across repetitions. Because it is a ratio normalized within each cell, it is more comparable across hardware than absolute tok/s.

This is useful for characterizing *workload-dependent* serving boundaries, e.g. long-decode traffic that stays stable up to high concurrency versus long-prefill or repeated-prefix traffic whose p95 latency amplifies much faster.

#### Collecting the data

Describe the workloads in a `--bench-params` JSON (one entry per workload, each with a unique `_benchmark_name`), and sweep concurrency via `--max-concurrency` in `--bench-cmd`. Keeping concurrency out of `--bench-params` lets `_benchmark_name` stay equal to the workload name, which is what the reducer groups by.

```json
[
    {
        "_benchmark_name": "balanced_2k",
        "dataset_name": "random",
        "random_input_len": 2048,
        "random_output_len": 128,
        "random_range_ratio": 0.0
    },
    {
        "_benchmark_name": "long_decode",
        "dataset_name": "random",
        "random_input_len": 1024,
        "random_output_len": 512,
        "random_range_ratio": 0.0
    },
    {
        "_benchmark_name": "long_prefill_8k",
        "dataset_name": "random",
        "random_input_len": 8192,
        "random_output_len": 64,
        "random_range_ratio": 0.0
    },
    {
        "_benchmark_name": "repeated_prefix",
        "dataset_name": "prefix_repetition",
        "prefix_repetition_prefix_len": 1792,
        "prefix_repetition_suffix_len": 256,
        "prefix_repetition_num_prefixes": 1,
        "prefix_repetition_output_len": 128
    }
]
```

```bash
MODEL=Qwen/Qwen2.5-7B-Instruct

for N in 1 2 4 8 16 32; do
    vllm bench sweep serve \
        --serve-cmd "vllm serve $MODEL --enable-prefix-caching" \
        --bench-cmd "vllm bench serve --model $MODEL --backend vllm --endpoint /v1/completions --num-prompts 320 --ignore-eos --max-concurrency $N --metric-percentiles 50,95,99" \
        --bench-params benchmarks/boundary_workloads.json \
        --num-runs 3 \
        --output-dir benchmarks/results/boundary/n$N \
        --experiment-name boundary
done
```

!!! important
    Pass `--metric-percentiles 50,95,99` inside `--bench-cmd`. `vllm bench sweep serve` sets `--percentile-metrics` but not `--metric-percentiles`, so the default leaves only `p99_*` columns and the reducer has no p95 to normalize.

!!! important
    For `repeated_prefix`, the shared prefix only benefits from the cache if prefix caching is on, so set `--enable-prefix-caching` on `--serve-cmd` and `--prefix-repetition-num-prefixes 1` so reuse dominates. Use a long, block-aligned prefix (a multiple of the KV block size, e.g. 1792) — a short prefix gives no measurable cache effect.

!!! note
    `--max-concurrency` caps the number of in-flight requests (open-loop), which differs from a closed-loop "N concurrent clients" model. Absolute tok/s and wall latency depend on hardware; the N=1-normalized multiplier is the portable signal. Use `--ignore-eos` so decode runs to the target output length.

#### Reducing to a table

Point `vllm bench sweep boundary` at the parent directory; it reads every `summary.json` beneath it.

```bash
vllm bench sweep boundary benchmarks/results/boundary
```

This writes `boundary.csv` and prints a table with, per `(workload, concurrency)`: averaged `output_throughput`, p50/p95 wall latency (`*_e2el_ms`), p50/p95 TTFT (`*_ttft_ms`), the `p95_e2el_ms_multiplier_vs_baseline` (and the TTFT equivalent), repetition count, and completed/failed counts.

To also sweep engine knobs, add a `--serve-params` JSON varying `max_num_batched_tokens`, `long_prefill_token_threshold`, and `enable_chunked_prefill`, then group by those columns too:

```bash
vllm bench sweep boundary benchmarks/results/boundary \
    --group-by _benchmark_name,max_num_batched_tokens,long_prefill_token_threshold,enable_chunked_prefill
```

!!! note
    When sweeping knobs, keep combinations self-consistent: disabling chunked prefill requires `max_num_batched_tokens >= max_model_len`, and `long_prefill_token_threshold` only takes effect when chunked prefill is enabled. Keep `--max-model-len` large enough for the `long_prefill_8k` prompts (≥ 8704).

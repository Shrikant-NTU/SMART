from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from pymcdm.methods import TOPSIS
except ImportError:  # pragma: no cover - handled at runtime with a clear message.
    TOPSIS = None


# =========================
# Easy-to-change GA settings
# =========================

INITIAL_POPULATION_SIZE = 100
WORKING_POPULATION_SIZE = 30
GENERATIONS = 35
CROSSOVER_PROBABILITY = 0.80
MUTATION_PROBABILITY = 0.15
RANDOM_SEED = 77
CALIBRATION_REPEAT_COUNT = 10
CALIBRATION_REPEAT_SEEDS = [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
TIMEOUT_SECONDS = 3600

PARALLEL_WORKERS = "all"
TERMINAL_VERBOSITY = "concise"
TERMINAL_GENERATION_INTERVAL = 5

PARAMETER_SPECS = {
    "tau": {"default": 0.5, "lower": 0.2, "upper": 1.2, "index": 3},
    "Ai": {"default": 2.1, "lower": 1.9, "upper": 4.0, "index": 4},
    "Bi": {"default": 0.3, "lower": 0.1, "upper": 0.6, "index": 6},
}

TOPSIS_METRICS = [
    "goal_reach_rate",
    "average_completion_time",
    "average_overlap_duration_seconds",
]

TOPSIS_WEIGHTS = {
    "goal_reach_rate": 0.25,
    "average_completion_time": 0.25,
    "average_overlap_duration_seconds": 0.50,
}

PARAMETER_MUTATION_SIZES = {
    "tau": 0.15,
    "Ai": 0.24,
    "Bi": 0.075,
}

METRIC_DIRECTIONS = {
    "goal_reach_rate": "maximize",
    "average_completion_time": "minimize",
    "average_overlap_duration_seconds": "minimize",
}

MIN_TOPSIS_FITNESS = 1e-12

FEASIBILITY_METRIC = "average_overlap_duration_seconds"
MAX_AVERAGE_OVERLAP_DURATION_SECONDS = 8.0
MAX_FEASIBILITY_EVALUATION_MULTIPLIER = 10
MAX_FEASIBILITY_REPLACEMENTS_TOTAL = 50

OBSERVED_METRICS = [
    "total_pedestrians",
    "average_speed",
    "stuck_pedestrian_count",
    "overlap_count",
    "unique_overlap_count",
    "overlap_episode_count",
    "total_overlap_duration_seconds",
]


SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_SCRIPT_NAME = "example10.py"
PARAMETER_FILE = Path("jsfm") / "utils.py"
METRICS_FILE = Path(".images") / "example10_metrics.json"


@dataclass
class CandidateSpec:
    candidate_id: str
    generation: int
    parameters: dict[str, float | int]


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def log(message: str = "", detail: bool = False) -> None:
    if detail and TERMINAL_VERBOSITY != "detailed":
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    text = str(message)
    if not text:
        print(f"[{timestamp}]")
        return
    for line in text.splitlines():
        print(f"[{timestamp}] {line}" if line else f"[{timestamp}]")


def log_generation(generation: int, message: str) -> None:
    visible = (
        generation == 1
        or generation == GENERATIONS
        or generation % max(1, TERMINAL_GENERATION_INTERVAL) == 0
    )
    log(message, detail=not visible)


def parameter_names() -> list[str]:
    return list(PARAMETER_SPECS)


def vector_to_parameters(vector: np.ndarray) -> dict[str, float | int]:
    parameters: dict[str, float | int] = {}
    for index, name in enumerate(parameter_names()):
        value = float(vector[index])
        parameters[name] = int(round(value)) if PARAMETER_SPECS[name].get("data_type") == "int" else value
    return parameters


def parameters_to_vector(parameters: dict[str, float]) -> np.ndarray:
    return np.array([float(parameters[name]) for name in parameter_names()], dtype=float)


def parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.array([PARAMETER_SPECS[name]["lower"] for name in parameter_names()], dtype=float)
    upper = np.array([PARAMETER_SPECS[name]["upper"] for name in parameter_names()], dtype=float)
    return lower, upper


def parameter_gene_space() -> list[dict[str, float]]:
    return [
        {"low": float(PARAMETER_SPECS[name]["lower"]), "high": float(PARAMETER_SPECS[name]["upper"])}
        for name in parameter_names()
    ]


def resolve_workers(candidate_count: int) -> int:
    cpu_count = os.cpu_count() or 1
    if PARALLEL_WORKERS == "all":
        return max(1, min(cpu_count, candidate_count))
    return max(1, min(int(PARALLEL_WORKERS), candidate_count))


def copy_source_workspace(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(SCRIPT_DIR / EXAMPLE_SCRIPT_NAME, destination / EXAMPLE_SCRIPT_NAME)
    jsfm_destination = destination / "jsfm"
    jsfm_destination.mkdir()
    for filename in ["__init__.py", "sfm.py", "metrics.py", "utils.py"]:
        shutil.copy2(SCRIPT_DIR / "jsfm" / filename, jsfm_destination / filename)


def patch_parameter_file(workspace: Path, parameters: dict[str, float]) -> None:
    parameter_path = workspace / PARAMETER_FILE
    text = parameter_path.read_text(encoding="utf-8")

    marker = "single_params = jnp.array(["
    start = text.find(marker)
    if start == -1:
        raise ValueError(f"Could not find single_params array in {parameter_path}")

    array_start = text.find("[", start)
    array_end = text.find("])", array_start)
    if array_start == -1 or array_end == -1:
        raise ValueError(f"Could not parse single_params array in {parameter_path}")

    body = text[array_start + 1 : array_end]
    lines = body.splitlines()
    value_line_indices = [
        index
        for index, line in enumerate(lines)
        if line.strip() and not line.strip().startswith("#")
    ]

    required_count = max(spec["index"] for spec in PARAMETER_SPECS.values()) + 1
    if len(value_line_indices) < required_count:
        raise ValueError(
            f"single_params has {len(value_line_indices)} values; expected at least {required_count}"
        )

    for name, value in parameters.items():
        param_index = PARAMETER_SPECS[name]["index"]
        line_index = value_line_indices[param_index]
        original_line = lines[line_index]
        indent = original_line[: len(original_line) - len(original_line.lstrip())]
        lines[line_index] = f"{indent}{value:.6f},"

    patched_body = "\n".join(lines)
    patched_text = text[: array_start + 1] + patched_body + text[array_end:]
    parameter_path.write_text(patched_text, encoding="utf-8")


def run_example(
    workspace: Path,
    save_media: bool,
    repeat_seed: int | None = None,
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "SFM_SAVE_MEDIA": "1" if save_media else "0",
            "SFM_SAVE_PNG": "1" if save_media else "0",
            "SFM_SAVE_MP4": "1" if save_media else "0",
            "SFM_CALIBRATION_METRICS_ONLY": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
        }
    )
    if repeat_seed is not None:
        env["SFM_RANDOM_SEED"] = str(int(repeat_seed))

    start = time.perf_counter()
    result = subprocess.run(
        [sys.executable, EXAMPLE_SCRIPT_NAME],
        cwd=workspace,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=TIMEOUT_SECONDS,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "runtime_seconds": time.perf_counter() - start,
    }


def enable_workspace_repeat_seed(workspace: Path) -> None:
    script_path = workspace / EXAMPLE_SCRIPT_NAME
    if not script_path.exists():
        return
    text = script_path.read_text(encoding="utf-8")
    seeded_call = 'np.random.default_rng(int(os.environ.get("SFM_RANDOM_SEED", "77")))'
    updated = text.replace("np.random.default_rng(77)", seeded_call)
    if updated != text:
        script_path.write_text(updated, encoding="utf-8")


def calibration_repeat_seeds() -> list[int | None]:
    repeat_count = max(1, int(CALIBRATION_REPEAT_COUNT))
    if repeat_count == 1:
        return [CALIBRATION_REPEAT_SEEDS[0] if CALIBRATION_REPEAT_SEEDS else None]
    if len(CALIBRATION_REPEAT_SEEDS) < repeat_count:
        raise ValueError(
            "CALIBRATION_REPEAT_SEEDS must contain at least "
            f"{repeat_count} seed(s); got {len(CALIBRATION_REPEAT_SEEDS)}."
        )
    return [int(seed) for seed in CALIBRATION_REPEAT_SEEDS[:repeat_count]]


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def aggregate_metric_values(values: list[Any]) -> Any:
    present = [value for value in values if value is not None]
    if not present:
        return None
    if all(_is_numeric(value) for value in present):
        return float(np.mean([float(value) for value in present]))
    if all(isinstance(value, dict) for value in present):
        keys = sorted({key for value in present for key in value})
        return {
            key: aggregate_metric_values([value.get(key) for value in present])
            for key in keys
        }
    if all(isinstance(value, list) for value in present):
        lengths = {len(value) for value in present}
        if len(lengths) == 1:
            length = lengths.pop()
            return [
                aggregate_metric_values([value[index] for value in present])
                for index in range(length)
            ]
    return present[0]


def aggregate_raw_metrics(raw_metrics_by_repeat: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for metrics in raw_metrics_by_repeat for key in metrics})
    aggregated = {
        key: aggregate_metric_values([metrics.get(key) for metrics in raw_metrics_by_repeat])
        for key in keys
    }
    aggregated["calibration_metric_aggregation"] = {
        "method": "mean_over_repeats",
        "repeat_count": len(raw_metrics_by_repeat),
        "repeat_seeds": calibration_repeat_seeds(),
    }
    return aggregated


def prepare_candidate_workspace(workspace: Path, parameters: dict[str, float | int]) -> None:
    if workspace.exists():
        shutil.rmtree(workspace)
    copy_source_workspace(workspace)
    enable_workspace_repeat_seed(workspace)
    patch_parameter_file(workspace, parameters)


def read_metrics(workspace: Path) -> dict[str, Any]:
    metrics_path = workspace / METRICS_FILE
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file was not created: {metrics_path}")
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def example10_metric_source(raw_metrics: dict[str, Any]) -> tuple[dict[str, Any], str]:
    merged = dict(raw_metrics)
    scenario_metrics = raw_metrics.get("scenario_metrics")
    if isinstance(scenario_metrics, list) and scenario_metrics:
        first = scenario_metrics[0]
        if isinstance(first, dict):
            merged.update(first)
            return merged, "example10_scenario_metrics"
    return merged, "example10_raw_metrics"


def extract_calibration_record(raw_metrics: dict[str, Any]) -> dict[str, Any]:
    source, source_name = example10_metric_source(raw_metrics)
    topsis_metrics = {}
    for metric in TOPSIS_METRICS:
        value = source.get(metric)
        if value is None and metric == "average_completion_time":
            value = source.get("completion_time")
        topsis_metrics[metric] = finite_float(value, default_for_metric(metric))

    observed_metrics = {metric: source.get(metric) for metric in OBSERVED_METRICS}

    return {
        "metric_format": "example10_archive_ga_cumulative_topsis_v1",
        "metric_source": source_name,
        "topsis_metrics": topsis_metrics,
        "observed_metrics": observed_metrics,
    }


def finite_float(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(numeric):
        return default
    return numeric


def default_for_metric(metric: str) -> float:
    if METRIC_DIRECTIONS[metric] == "maximize":
        return 0.0
    if metric == "average_completion_time":
        return TIMEOUT_SECONDS
    return 1_000_000.0


def round_dict(values: dict[str, float], places: int = 6) -> dict[str, float | int]:
    return {
        name: (
            int(round(float(value)))
            if PARAMETER_SPECS.get(name, {}).get("data_type") == "int"
            else round(float(value), places)
        )
        for name, value in values.items()
    }


def vector_key(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(round(float(value), 6) for value in vector)


def failure_metrics() -> dict[str, float]:
    return {metric: default_for_metric(metric) for metric in TOPSIS_METRICS}


def failure_calibration_record() -> dict[str, Any]:
    return {
        "metric_format": "example10_archive_ga_cumulative_topsis_v1",
        "metric_source": "failure_defaults",
        "topsis_metrics": failure_metrics(),
        "observed_metrics": {metric: None for metric in OBSERVED_METRICS},
    }


def label_candidate_feasibility(record: dict[str, Any]) -> bool:
    raw_value = record.get("topsis_metrics", {}).get(FEASIBILITY_METRIC)
    try:
        metric_value = float(raw_value)
    except (TypeError, ValueError):
        metric_value = math.inf

    is_feasible = math.isfinite(metric_value) and metric_value <= MAX_AVERAGE_OVERLAP_DURATION_SECONDS
    record["is_feasible"] = is_feasible
    record["feasibility_status"] = "feasible" if is_feasible else "not_feasible"
    record["removed_from_candidate_pool"] = not is_feasible
    record["feasibility_constraint"] = {
        "metric": FEASIBILITY_METRIC,
        "operator": "<=",
        "threshold": MAX_AVERAGE_OVERLAP_DURATION_SECONDS,
        "observed_value": raw_value,
    }
    if not is_feasible:
        if math.isfinite(metric_value):
            reason = (
                f"{FEASIBILITY_METRIC}={metric_value} exceeds "
                f"the maximum {MAX_AVERAGE_OVERLAP_DURATION_SECONDS}"
            )
        else:
            reason = f"{FEASIBILITY_METRIC} is missing or non-finite"
        record["feasibility_reason"] = reason
    else:
        record.pop("feasibility_reason", None)
    return is_feasible


def topsis_arrays() -> tuple[np.ndarray, np.ndarray]:
    weights = np.array([TOPSIS_WEIGHTS[metric] for metric in TOPSIS_METRICS], dtype=float)
    weights = weights / np.sum(weights)
    types = np.array(
        [1 if METRIC_DIRECTIONS[metric] == "maximize" else -1 for metric in TOPSIS_METRICS],
        dtype=int,
    )
    return weights, types


def compute_library_topsis_scores(records: list[dict[str, Any]]) -> list[float]:
    if TOPSIS is None:
        raise RuntimeError("pymcdm is required for GA TOPSIS scoring. Install it with: python -m pip install pymcdm")
    if not records:
        return []
    if len(records) == 1:
        return [1.0]

    matrix = np.array(
        [
            [
                finite_float(record.get("topsis_metrics", {}).get(metric), default_for_metric(metric))
                for metric in TOPSIS_METRICS
            ]
            for record in records
        ],
        dtype=float,
    )
    weights, types = topsis_arrays()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scores = TOPSIS()(matrix, weights, types)
    return [float(score) if math.isfinite(float(score)) else MIN_TOPSIS_FITNESS for score in scores]


def rescore_records_with_topsis(
    records: list[dict[str, Any]],
    score_key: str = "current_topsis_score",
) -> None:
    feasible_records = [record for record in records if record.get("is_feasible", False)]
    scores = compute_library_topsis_scores(feasible_records)
    ranked_indices = sorted(range(len(feasible_records)), key=lambda index: scores[index], reverse=True)
    ranks = {index: rank + 1 for rank, index in enumerate(ranked_indices)}

    for record in records:
        if record.get("is_feasible", False):
            continue
        record[score_key] = MIN_TOPSIS_FITNESS
        if score_key == "current_topsis_score":
            record["fitness"] = MIN_TOPSIS_FITNESS
            record["topsis_rank_current_archive"] = None

    for index, record in enumerate(feasible_records):
        score = max(float(scores[index]), MIN_TOPSIS_FITNESS)
        record[score_key] = round(score, 9)
        if score_key == "current_topsis_score":
            record["fitness"] = score
            record["topsis_rank_current_archive"] = ranks[index]


def evaluate_candidate(spec: CandidateSpec, run_dir: Path) -> dict[str, Any]:
    workspace = run_dir / "workspaces" / spec.candidate_id
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "candidate_id": spec.candidate_id,
        "generation": spec.generation,
        "parameters": round_dict(spec.parameters),
        "workspace": str(workspace),
        "success": False,
    }

    try:
        if workspace.exists():
            shutil.rmtree(workspace)
        copy_source_workspace(workspace)
        enable_workspace_repeat_seed(workspace)
        patch_parameter_file(workspace, spec.parameters)

        repeat_raw_metrics: list[dict[str, Any]] = []
        repeat_records: list[dict[str, Any]] = []
        repeat_run_results: list[dict[str, Any]] = []
        repeat_seeds = calibration_repeat_seeds()
        for repeat_index, repeat_seed in enumerate(repeat_seeds, start=1):
            run_result = run_example(workspace, save_media=False, repeat_seed=repeat_seed)
            repeat_label = f"{spec.candidate_id}_repeat{repeat_index:02d}"
            (logs_dir / f"{repeat_label}_stdout.txt").write_text(run_result["stdout"], encoding="utf-8")
            (logs_dir / f"{repeat_label}_stderr.txt").write_text(run_result["stderr"], encoding="utf-8")
            repeat_result = {
                "repeat": repeat_index,
                "seed": repeat_seed,
                "returncode": run_result["returncode"],
                "runtime_seconds": round(run_result["runtime_seconds"], 4),
            }
            repeat_run_results.append(repeat_result)

            if run_result["returncode"] != 0:
                record["error"] = (
                    f"Repeat {repeat_index}/{len(repeat_seeds)} failed with return code "
                    f"{run_result['returncode']}"
                )
                record["run_result"] = {
                    "returncode": run_result["returncode"],
                    "runtime_seconds": round(sum(item["runtime_seconds"] for item in repeat_run_results), 4),
                    "repeat_count": len(repeat_seeds),
                    "repeat_success_count": len(repeat_raw_metrics),
                    "repeat_runs": repeat_run_results,
                }
                record.update(failure_calibration_record())
                record["fitness"] = MIN_TOPSIS_FITNESS
                return record

            raw_metrics = read_metrics(workspace)
            repeat_raw_metrics.append(raw_metrics)
            repeat_record = extract_calibration_record(raw_metrics)
            repeat_record["repeat"] = repeat_index
            repeat_record["seed"] = repeat_seed
            repeat_records.append(repeat_record)
            (logs_dir / f"{repeat_label}_metrics.json").write_text(
                json.dumps(raw_metrics, indent=2),
                encoding="utf-8",
            )

        aggregated_metrics = aggregate_raw_metrics(repeat_raw_metrics)
        metrics_path = workspace / METRICS_FILE
        metrics_path.write_text(json.dumps(aggregated_metrics, indent=2), encoding="utf-8")
        record["raw_metrics_path"] = str(metrics_path)
        record["calibration_repeat_count"] = len(repeat_seeds)
        record["calibration_repeat_seeds"] = repeat_seeds
        record["repeat_metrics"] = repeat_records
        record["run_result"] = {
            "returncode": 0,
            "runtime_seconds": round(sum(item["runtime_seconds"] for item in repeat_run_results), 4),
            "mean_runtime_seconds": round(
                sum(item["runtime_seconds"] for item in repeat_run_results) / len(repeat_run_results),
                4,
            ),
            "repeat_count": len(repeat_seeds),
            "repeat_success_count": len(repeat_raw_metrics),
            "repeat_runs": repeat_run_results,
        }
        record.update(extract_calibration_record(aggregated_metrics))
        record["fitness"] = MIN_TOPSIS_FITNESS
        record["success"] = True
        return record
    except subprocess.TimeoutExpired:
        record["error"] = f"Simulation timed out after {TIMEOUT_SECONDS} seconds"
        record.update(failure_calibration_record())
        record["fitness"] = MIN_TOPSIS_FITNESS
        return record
    except Exception as exc:
            record["error"] = str(exc)
            record.update(failure_calibration_record())
            record["fitness"] = MIN_TOPSIS_FITNESS
            return record


def candidate_failure_record(
    spec: CandidateSpec,
    workspace: Path,
    error: str,
    repeat_run_results: list[dict[str, Any]] | None = None,
    repeat_success_count: int = 0,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "candidate_id": spec.candidate_id,
        "generation": spec.generation,
        "parameters": round_dict(spec.parameters),
        "workspace": str(workspace),
        "success": False,
        "error": error,
        "run_result": {
            "returncode": 1,
            "runtime_seconds": round(
                sum(float(item.get("runtime_seconds", 0.0)) for item in (repeat_run_results or [])),
                4,
            ),
            "repeat_count": len(calibration_repeat_seeds()),
            "repeat_success_count": repeat_success_count,
            "repeat_runs": repeat_run_results or [],
        },
    }
    record.update(failure_calibration_record())
    record["fitness"] = MIN_TOPSIS_FITNESS
    return record


def run_candidate_repeat(
    spec: CandidateSpec,
    run_dir: Path,
    repeat_index: int,
    repeat_seed: int | None,
) -> dict[str, Any]:
    workspace = run_dir / "workspaces" / spec.candidate_id
    repeat_workspace = workspace / "repeat_workspaces" / f"repeat_{repeat_index:02d}_seed_{repeat_seed}"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if repeat_workspace.exists():
        shutil.rmtree(repeat_workspace)
    copy_source_workspace(repeat_workspace)
    enable_workspace_repeat_seed(repeat_workspace)
    patch_parameter_file(repeat_workspace, spec.parameters)

    run_result = run_example(repeat_workspace, save_media=False, repeat_seed=repeat_seed)
    repeat_label = f"{spec.candidate_id}_repeat{repeat_index:02d}"
    (logs_dir / f"{repeat_label}_stdout.txt").write_text(run_result["stdout"], encoding="utf-8")
    (logs_dir / f"{repeat_label}_stderr.txt").write_text(run_result["stderr"], encoding="utf-8")
    repeat_result = {
        "repeat": repeat_index,
        "seed": repeat_seed,
        "returncode": run_result["returncode"],
        "runtime_seconds": round(run_result["runtime_seconds"], 4),
        "workspace": str(repeat_workspace),
    }

    if run_result["returncode"] != 0:
        return {
            "repeat_result": repeat_result,
            "raw_metrics": None,
            "repeat_record": None,
            "error": (
                f"Repeat {repeat_index}/{len(calibration_repeat_seeds())} failed with return code "
                f"{run_result['returncode']}"
            ),
        }

    raw_metrics = read_metrics(repeat_workspace)
    repeat_record = extract_calibration_record(raw_metrics)
    repeat_record["repeat"] = repeat_index
    repeat_record["seed"] = repeat_seed
    repeat_record["workspace"] = str(repeat_workspace)
    (logs_dir / f"{repeat_label}_metrics.json").write_text(
        json.dumps(raw_metrics, indent=2),
        encoding="utf-8",
    )
    return {
        "repeat_result": repeat_result,
        "raw_metrics": raw_metrics,
        "repeat_record": repeat_record,
        "error": None,
    }


def assemble_candidate_record(
    spec: CandidateSpec,
    run_dir: Path,
    repeat_outputs: list[dict[str, Any]],
) -> dict[str, Any]:
    workspace = run_dir / "workspaces" / spec.candidate_id
    repeat_seeds = calibration_repeat_seeds()
    repeat_outputs = sorted(repeat_outputs, key=lambda item: item["repeat_result"]["repeat"])
    repeat_run_results = [item["repeat_result"] for item in repeat_outputs]
    repeat_raw_metrics = [
        item["raw_metrics"]
        for item in repeat_outputs
        if item.get("raw_metrics") is not None
    ]
    repeat_records = [
        item["repeat_record"]
        for item in repeat_outputs
        if item.get("repeat_record") is not None
    ]
    errors = [item["error"] for item in repeat_outputs if item.get("error")]

    if errors:
        return candidate_failure_record(
            spec,
            workspace,
            str(errors[0]),
            repeat_run_results=repeat_run_results,
            repeat_success_count=len(repeat_raw_metrics),
        )

    try:
        if not (workspace / EXAMPLE_SCRIPT_NAME).exists():
            prepare_candidate_workspace(workspace, spec.parameters)
        aggregated_metrics = aggregate_raw_metrics(repeat_raw_metrics)
        metrics_path = workspace / METRICS_FILE
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(aggregated_metrics, indent=2), encoding="utf-8")

        record: dict[str, Any] = {
            "candidate_id": spec.candidate_id,
            "generation": spec.generation,
            "parameters": round_dict(spec.parameters),
            "workspace": str(workspace),
            "success": True,
            "raw_metrics_path": str(metrics_path),
            "calibration_repeat_count": len(repeat_seeds),
            "calibration_repeat_seeds": repeat_seeds,
            "repeat_metrics": repeat_records,
            "run_result": {
                "returncode": 0,
                "runtime_seconds": round(
                    sum(float(item.get("runtime_seconds", 0.0)) for item in repeat_run_results),
                    4,
                ),
                "mean_runtime_seconds": round(
                    sum(float(item.get("runtime_seconds", 0.0)) for item in repeat_run_results)
                    / len(repeat_run_results),
                    4,
                ),
                "repeat_count": len(repeat_seeds),
                "repeat_success_count": len(repeat_raw_metrics),
                "repeat_runs": repeat_run_results,
            },
        }
        record.update(extract_calibration_record(aggregated_metrics))
        record["fitness"] = MIN_TOPSIS_FITNESS
        return record
    except Exception as exc:  # noqa: BLE001 - convert candidate-level setup failures to GA failures.
        return candidate_failure_record(
            spec,
            workspace,
            str(exc),
            repeat_run_results=repeat_run_results,
            repeat_success_count=len(repeat_raw_metrics),
        )


def add_generation_topsis_summary(
    records: list[dict[str, Any]],
    archive_records: list[dict[str, Any]],
) -> dict[str, Any]:
    feasible_records = [record for record in records if record.get("is_feasible", False)]
    removed_records = [record for record in records if not record.get("is_feasible", False)]
    scores = [float(record.get("current_topsis_score", MIN_TOPSIS_FITNESS)) for record in feasible_records]
    ranked_indices = sorted(range(len(feasible_records)), key=lambda index: scores[index], reverse=True)
    ranks = {index: rank + 1 for rank, index in enumerate(ranked_indices)}

    for record in removed_records:
        record["generation_topsis_rank"] = {
            "rank_within_generation": None,
            "ranking_metric": "current_topsis_score",
            "higher_is_better": True,
            "excluded_by_feasibility_constraint": True,
        }
    for index, record in enumerate(feasible_records):
        record["generation_topsis_rank"] = {
            "rank_within_generation": ranks[index],
            "ranking_metric": "current_topsis_score",
            "higher_is_better": True,
            "excluded_by_feasibility_constraint": False,
        }

    best = feasible_records[ranked_indices[0]] if ranked_indices else None
    archive_best = max(
        archive_records,
        key=lambda record: float(record.get("current_topsis_score", MIN_TOPSIS_FITNESS)),
        default=None,
    )
    average_score = round(float(np.mean(scores)), 9) if scores else MIN_TOPSIS_FITNESS
    return {
        "metrics": list(TOPSIS_METRICS),
        "topsis_weights": TOPSIS_WEIGHTS,
        "directions": METRIC_DIRECTIONS,
        "scoring": "cumulative_pymcdm_topsis_feasible_candidates_only",
        "candidate_count": len(records),
        "feasible_candidate_count": len(feasible_records),
        "removed_candidate_count": len(removed_records),
        "removed_candidate_ids": [record["candidate_id"] for record in removed_records],
        "average_topsis_score": average_score,
        "best_candidate_id": best["candidate_id"] if best else None,
        "best_candidate_topsis_score_at_time": best.get("current_topsis_score") if best else None,
        "best_candidate_topsis_metrics": best["topsis_metrics"] if best else None,
        "best_candidate_observed_metrics": best["observed_metrics"] if best else None,
        "best_candidate_parameters": best["parameters"] if best else None,
        "best_so_far_candidate_id": archive_best["candidate_id"] if archive_best else None,
        "best_so_far_topsis_score_at_time": archive_best.get("current_topsis_score") if archive_best else None,
        "best_so_far_generation": archive_best.get("generation") if archive_best else None,
        "best_so_far_topsis_metrics": archive_best.get("topsis_metrics") if archive_best else None,
        "topsis_reference": "pymcdm TOPSIS with ideal best/worst derived from feasible candidates evaluated so far",
    }


def write_generation_csv(records: list[dict[str, Any]], csv_path: Path) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "source_generation",
                "generation_candidate_role",
                "feasibility_replacement_batch",
                "feasibility_status",
                "removed_from_candidate_pool",
                "feasibility_reason",
                *parameter_names(),
                "fitness",
                "current_topsis_score",
                "final_topsis_score",
                "average_overlap_duration",
                "goal_reach_rate",
                "average_completion_time",
                "generation_topsis_rank",
                "archive_topsis_rank",
            ],
        )
        writer.writeheader()
        for record in records:
            metrics = record["topsis_metrics"]
            writer.writerow(
                {
                    "candidate_id": record["candidate_id"],
                    "source_generation": record["generation"],
                    "generation_candidate_role": record.get("generation_candidate_role"),
                    "feasibility_replacement_batch": record.get("feasibility_replacement_batch"),
                    "feasibility_status": record.get("feasibility_status"),
                    "removed_from_candidate_pool": record.get("removed_from_candidate_pool"),
                    "feasibility_reason": record.get("feasibility_reason"),
                    **record["parameters"],
                    "fitness": record.get("fitness"),
                    "current_topsis_score": record.get("current_topsis_score"),
                    "final_topsis_score": record.get("final_topsis_score"),
                    "average_overlap_duration": metrics.get("average_overlap_duration_seconds"),
                    "goal_reach_rate": metrics.get("goal_reach_rate"),
                    "average_completion_time": metrics.get("average_completion_time"),
                    "generation_topsis_rank": record.get("generation_topsis_rank", {}).get("rank_within_generation"),
                    "archive_topsis_rank": record.get("topsis_rank_current_archive"),
                }
            )


def save_removed_candidate_records(state: CalibrationState) -> None:
    (state.run_dir / "removed_candidates.json").write_text(
        json.dumps(state.removed_records, indent=2),
        encoding="utf-8",
    )


def exit_for_insufficient_feasible_candidates(
    state: CalibrationState,
    generation: int,
    target_count: int,
    feasible_count: int,
    evaluated_count: int,
    maximum_evaluation_count: int,
    phase: str,
) -> None:
    details = {
        "status": "stopped_insufficient_feasible_candidates",
        "generation": generation,
        "phase": phase,
        "required_feasible_candidates": target_count,
        "feasible_candidates_found": feasible_count,
        "candidates_evaluated": evaluated_count,
        "candidates_removed_in_generation": evaluated_count - feasible_count,
        "feasibility_metric": FEASIBILITY_METRIC,
        "feasibility_threshold": MAX_AVERAGE_OVERLAP_DURATION_SECONDS,
        "maximum_evaluation_multiplier": MAX_FEASIBILITY_EVALUATION_MULTIPLIER,
        "maximum_evaluation_count": maximum_evaluation_count,
        "maximum_feasibility_replacements_total": MAX_FEASIBILITY_REPLACEMENTS_TOTAL,
        "feasibility_replacements_evaluated": state.feasibility_replacements_evaluated,
    }
    save_removed_candidate_records(state)
    (state.run_dir / "all_records.json").write_text(
        json.dumps(state.all_records, indent=2),
        encoding="utf-8",
    )
    (state.run_dir / "calibration_stopped.json").write_text(
        json.dumps(details, indent=2),
        encoding="utf-8",
    )

    log("")
    log("Calibration stopped: the feasible population could not be filled.")
    log(
        f"Generation {generation} ({phase}) found {feasible_count}/{target_count} feasible "
        f"candidate(s) after {evaluated_count} evaluations."
    )
    log(
        f"Constraint: {FEASIBILITY_METRIC} <= {MAX_AVERAGE_OVERLAP_DURATION_SECONDS}. "
        f"Generation evaluation limit: {maximum_evaluation_count}; "
        f"total replacement limit: {MAX_FEASIBILITY_REPLACEMENTS_TOTAL}."
    )
    log(
        "Increase the configured feasibility replacement limit, relax the feasibility "
        "threshold, or adjust the parameter bounds before running again."
    )
    log(f"Partial results: {state.run_dir}")
    raise SystemExit(1)


def save_generation_outputs(records: list[dict[str, Any]], run_dir: Path, generation: int) -> dict[str, Any]:
    records = [dict(record) for record in records]
    records.sort(key=lambda record: float(record.get("current_topsis_score", MIN_TOPSIS_FITNESS)), reverse=True)
    state = active_state()
    summary = add_generation_topsis_summary(records, state.feasible_records)
    summary["generation"] = generation
    summary["target_feasible_candidate_count"] = WORKING_POPULATION_SIZE
    summary["feasibility_replacement_evaluation_count"] = sum(
        int(record.get("feasibility_replacement_batch", 0) > 0)
        for record in records
    )
    summary["feasibility_replacement_batch_count"] = max(
        (int(record.get("feasibility_replacement_batch", 0)) for record in records),
        default=0,
    )

    generation_dir = run_dir / f"generation_{generation:03d}"
    generation_dir.mkdir(parents=True, exist_ok=True)
    (generation_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (generation_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_generation_csv(records, generation_dir / "generation_metrics.csv")
    save_removed_candidate_records(state)
    return summary


class CalibrationState:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.lock = threading.Lock()
        self.next_candidate_number = 1
        self.records_by_key: dict[tuple[float, ...], dict[str, Any]] = {}
        self.all_records: list[dict[str, Any]] = []
        self.feasible_records: list[dict[str, Any]] = []
        self.removed_records: list[dict[str, Any]] = []
        self.generation_summaries: list[dict[str, Any]] = []
        self.feasibility_replacements_evaluated = 0

    def next_candidate_id(self) -> str:
        with self.lock:
            candidate_id = f"ga_c{self.next_candidate_number:04d}"
            self.next_candidate_number += 1
            return candidate_id

    def store_record(self, vector: np.ndarray, record: dict[str, Any]) -> None:
        is_feasible = label_candidate_feasibility(record)
        with self.lock:
            self.records_by_key[vector_key(vector)] = record
            self.all_records.append(record)
            if is_feasible:
                self.feasible_records.append(record)
            else:
                self.removed_records.append(record)

    def get_record(self, vector: np.ndarray) -> dict[str, Any] | None:
        with self.lock:
            return self.records_by_key.get(vector_key(vector))

    def evaluate_vector(self, vector: np.ndarray, generation: int) -> dict[str, Any]:
        vector = clipped_vector(vector)
        cached = self.get_record(vector)
        if cached is not None:
            return cached

        spec = CandidateSpec(
            candidate_id=self.next_candidate_id(),
            generation=generation,
            parameters=vector_to_parameters(vector),
        )
        record = evaluate_candidate(spec, self.run_dir)
        self.store_record(vector, record)
        return record

    def records_for_population(self, population: np.ndarray) -> list[dict[str, Any]]:
        records = []
        for vector in np.asarray(population, dtype=float):
            record = self.get_record(clipped_vector(vector))
            if record is None:
                raise RuntimeError(
                    "Could not match a final archive parent vector to a saved candidate record. "
                    "Final selection is restricted to the final archive parent pool."
                )
            records.append(record)
        return records


_ACTIVE_STATE: CalibrationState | None = None


def active_state() -> CalibrationState:
    if _ACTIVE_STATE is None:
        raise RuntimeError("Calibration state is not active.")
    return _ACTIVE_STATE


def remaining_feasibility_replacement_budget(state: CalibrationState) -> int:
    return max(
        0,
        MAX_FEASIBILITY_REPLACEMENTS_TOTAL - state.feasibility_replacements_evaluated,
    )


def reserve_feasibility_replacements(state: CalibrationState, requested_count: int) -> int:
    granted_count = min(
        max(0, int(requested_count)),
        remaining_feasibility_replacement_budget(state),
    )
    state.feasibility_replacements_evaluated += granted_count
    return granted_count


def clipped_vector(vector: np.ndarray) -> np.ndarray:
    lower, upper = parameter_bounds()
    clipped = np.clip(np.asarray(vector, dtype=float), lower, upper)
    for index, name in enumerate(parameter_names()):
        if PARAMETER_SPECS[name].get("data_type") == "int":
            clipped[index] = round(float(clipped[index]))
    return clipped


def generate_initial_population() -> np.ndarray:
    lower, upper = parameter_bounds()
    rng = np.random.default_rng(RANDOM_SEED)
    used_keys: set[tuple[float, ...]] = set()
    candidates: list[np.ndarray] = []
    while len(candidates) < INITIAL_POPULATION_SIZE:
        candidate = clipped_vector(rng.uniform(lower, upper))
        key = vector_key(candidate)
        if key in used_keys:
            continue
        used_keys.add(key)
        candidates.append(candidate)
    return np.asarray(candidates, dtype=float)


def generate_unique_random_candidates(
    state: CalibrationState,
    candidate_count: int,
    batch_index: int,
) -> np.ndarray:
    lower, upper = parameter_bounds()
    rng = np.random.default_rng(RANDOM_SEED + 100000 + batch_index)
    used_keys = {
        vector_key(parameters_to_vector(record["parameters"]))
        for record in state.all_records
    }
    candidates: list[np.ndarray] = []
    for _ in range(candidate_count):
        for _attempt in range(10000):
            candidate = clipped_vector(rng.uniform(lower, upper))
            key = vector_key(candidate)
            if key in used_keys:
                continue
            used_keys.add(key)
            candidates.append(candidate)
            break
        else:
            raise RuntimeError("Could not generate a unique random feasibility-replacement candidate.")
    return np.asarray(candidates, dtype=float)


def evaluate_specs_parallel(
    state: CalibrationState,
    specs_and_vectors: list[tuple[CandidateSpec, np.ndarray]],
    generation: int,
    label: str,
) -> list[tuple[CandidateSpec, np.ndarray, dict[str, Any]]]:
    repeat_seeds = calibration_repeat_seeds()
    preparation_failures: dict[str, dict[str, Any]] = {}
    prepared_specs_and_vectors: list[tuple[CandidateSpec, np.ndarray]] = []
    for spec, vector in specs_and_vectors:
        workspace = state.run_dir / "workspaces" / spec.candidate_id
        try:
            prepare_candidate_workspace(workspace, spec.parameters)
            prepared_specs_and_vectors.append((spec, vector))
        except Exception as exc:  # noqa: BLE001 - keep the GA generation moving.
            preparation_failures[spec.candidate_id] = candidate_failure_record(
                spec,
                workspace,
                str(exc),
                repeat_run_results=[],
                repeat_success_count=0,
            )

    repeat_jobs = [
        (spec, vector, repeat_index, repeat_seed)
        for repeat_index, repeat_seed in enumerate(repeat_seeds, start=1)
        for spec, vector in prepared_specs_and_vectors
    ]
    workers = resolve_workers(len(repeat_jobs))
    log(
        f"Generation {generation}: evaluating {len(specs_and_vectors)} {label} candidate(s) "
        f"with {len(repeat_seeds)} repeat(s) each ({len(repeat_jobs)} simulation job(s)) "
        f"on {workers} worker(s)",
        detail=True,
    )

    repeat_outputs_by_candidate: dict[str, list[dict[str, Any]]] = {
        spec.candidate_id: []
        for spec, _vector in prepared_specs_and_vectors
    }
    if repeat_jobs:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(
                    run_candidate_repeat,
                    spec,
                    state.run_dir,
                    repeat_index,
                    repeat_seed,
                ): (spec, vector, repeat_index, repeat_seed)
                for spec, vector, repeat_index, repeat_seed in repeat_jobs
            }
            completed = 0
            for future in as_completed(future_to_item):
                spec, _vector, repeat_index, _repeat_seed = future_to_item[future]
                completed += 1
                try:
                    repeat_output = future.result()
                except Exception as exc:  # noqa: BLE001 - keep failed candidates in GA accounting.
                    repeat_output = {
                        "repeat_result": {
                            "repeat": repeat_index,
                            "seed": _repeat_seed,
                            "returncode": 1,
                            "runtime_seconds": 0.0,
                        },
                        "raw_metrics": None,
                        "repeat_record": None,
                        "error": str(exc),
                    }
                repeat_outputs_by_candidate[spec.candidate_id].append(repeat_output)
                status = "done" if repeat_output.get("error") is None else "failed"
                log(
                    f"Generation {generation}: {spec.candidate_id} repeat {repeat_index} {status} "
                    f"({completed}/{len(repeat_jobs)})",
                    detail=True,
                )

    results = []
    for spec, vector in specs_and_vectors:
        if spec.candidate_id in preparation_failures:
            record = preparation_failures[spec.candidate_id]
        else:
            record = assemble_candidate_record(
                spec,
                state.run_dir,
                repeat_outputs_by_candidate[spec.candidate_id],
            )
        state.store_record(vector, record)
        results.append((spec, vector, record))
        status = "done" if record.get("success") else "failed"
        log(
            f"Generation {generation}: {spec.candidate_id} aggregated {status}",
            detail=True,
        )
    return results


def evaluate_generation_vectors(
    state: CalibrationState,
    vectors: np.ndarray,
    generation: int,
    label: str,
) -> list[dict[str, Any]]:
    normalized_vectors = [clipped_vector(vector) for vector in np.asarray(vectors, dtype=float)]
    specs_and_vectors = [
        (
            CandidateSpec(
                candidate_id=state.next_candidate_id(),
                generation=generation,
                parameters=vector_to_parameters(vector),
            ),
            vector,
        )
        for vector in normalized_vectors
    ]
    results = evaluate_specs_parallel(state, specs_and_vectors, generation, label)
    return [record for _, _, record in results]


def evaluate_initial_population(
    state: CalibrationState,
    initial_population: np.ndarray,
    generation: int,
) -> list[dict[str, Any]]:
    records = evaluate_generation_vectors(state, initial_population, generation, "initial")
    for record in records:
        record["generation_candidate_role"] = "initial_random"
        record["feasibility_replacement_batch"] = 0

    replacement_batch = 0
    max_evaluations = INITIAL_POPULATION_SIZE * MAX_FEASIBILITY_EVALUATION_MULTIPLIER
    feasible_count = sum(record.get("is_feasible", False) for record in records)
    while feasible_count < WORKING_POPULATION_SIZE:
        remaining_budget = max_evaluations - len(records)
        remaining_replacement_budget = remaining_feasibility_replacement_budget(state)
        if remaining_budget <= 0 or remaining_replacement_budget <= 0:
            exit_for_insufficient_feasible_candidates(
                state=state,
                generation=generation,
                target_count=WORKING_POPULATION_SIZE,
                feasible_count=feasible_count,
                evaluated_count=len(records),
                maximum_evaluation_count=max_evaluations,
                phase="initial random population",
            )

        replacement_batch += 1
        missing_count = WORKING_POPULATION_SIZE - feasible_count
        replacement_count = reserve_feasibility_replacements(
            state,
            min(missing_count, remaining_budget),
        )
        replacement_vectors = generate_unique_random_candidates(
            state,
            replacement_count,
            replacement_batch,
        )
        replacements = evaluate_generation_vectors(
            state,
            replacement_vectors,
            generation,
            "initial-feasibility-replacement",
        )
        for record in replacements:
            record["generation_candidate_role"] = "initial_random_feasibility_replacement"
            record["feasibility_replacement_batch"] = replacement_batch
        records.extend(replacements)
        feasible_count += sum(record.get("is_feasible", False) for record in replacements)
        save_removed_candidate_records(state)
        log(
            f"Generation {generation}: initial feasibility replacement batch {replacement_batch} "
            f"retained {sum(record.get('is_feasible', False) for record in replacements)}/"
            f"{len(replacements)} candidate(s); {WORKING_POPULATION_SIZE - feasible_count} "
            "working-population slot(s) remain"
        )

    rescore_records_with_topsis(state.all_records)

    summary = save_generation_outputs(records, state.run_dir, generation)
    state.generation_summaries.append(summary)
    return records


def select_archive_parent_records(state: CalibrationState) -> list[dict[str, Any]]:
    rescore_records_with_topsis(state.feasible_records)
    ranked = sorted(
        state.feasible_records,
        key=lambda record: float(record.get("current_topsis_score", MIN_TOPSIS_FITNESS)),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    used_keys: set[tuple[float, ...]] = set()
    for record in ranked:
        key = vector_key(parameters_to_vector(record["parameters"]))
        if key in used_keys:
            continue
        selected.append(record)
        used_keys.add(key)
        if len(selected) == WORKING_POPULATION_SIZE:
            break
    if len(selected) < 2:
        raise RuntimeError(
            f"TOPSIS archive parent selection found only {len(selected)} feasible unique candidate(s); "
            "at least 2 are required for crossover."
        )
    if len(selected) < WORKING_POPULATION_SIZE:
        log(
            f"Feasibility filtering left {len(selected)} archive parent(s), fewer than the requested "
            f"{WORKING_POPULATION_SIZE}; child generation will use the available feasible parents."
        )
    return selected


def parent_records_to_population(parent_records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([parameters_to_vector(record["parameters"]) for record in parent_records], dtype=float)


def mutate_child_vector(vector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    child = np.asarray(vector, dtype=float).copy()
    step_sizes = np.array([PARAMETER_MUTATION_SIZES[name] for name in parameter_names()], dtype=float)
    mask = rng.random(child.shape) < MUTATION_PROBABILITY
    deltas = rng.uniform(-step_sizes, step_sizes, size=child.shape)
    child = np.where(mask, child + deltas, child)
    return clipped_vector(child)


def crossover_parent_vectors(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    parent_a = np.asarray(parent_a, dtype=float)
    parent_b = np.asarray(parent_b, dtype=float)
    if rng.random() > CROSSOVER_PROBABILITY:
        return parent_a.copy()

    child = parent_a.copy()
    if len(child) == 1:
        child[0] = parent_a[0] if rng.random() < 0.5 else parent_b[0]
        return child

    crossover_point = int(rng.integers(1, len(child)))
    child[crossover_point:] = parent_b[crossover_point:]
    return child


def make_unique_child_from_parents(
    ranked_parent_vectors: list[np.ndarray],
    parent_pair_index: int,
    rng: np.random.Generator,
    used_keys: set[tuple[float, ...]],
) -> np.ndarray:
    if len(ranked_parent_vectors) < 2:
        raise RuntimeError("Need at least two parent candidates to repair duplicate GA slots.")

    parent_count = len(ranked_parent_vectors)
    for attempt in range(10000):
        start = (parent_pair_index + attempt) % parent_count
        parent_a = ranked_parent_vectors[start]
        parent_b = ranked_parent_vectors[(start + 1) % parent_count]
        child = crossover_parent_vectors(parent_a, parent_b, rng)
        child = mutate_child_vector(child, rng)
        key = vector_key(child)
        if key not in used_keys:
            used_keys.add(key)
            return child

    raise RuntimeError("Could not generate a unique GA repair child after 10000 attempts.")


def generate_archive_children(
    parent_records: list[dict[str, Any]],
    state: CalibrationState,
    generation: int,
    candidate_count: int = WORKING_POPULATION_SIZE,
    batch_index: int = 0,
) -> np.ndarray:
    ranked_parent_vectors = [
        parameters_to_vector(record["parameters"])
        for record in sorted(
            parent_records,
            key=lambda item: float(item.get("current_topsis_score", MIN_TOPSIS_FITNESS)),
            reverse=True,
        )
    ]
    used_keys = {
        vector_key(parameters_to_vector(record["parameters"]))
        for record in state.all_records
    }
    children: list[np.ndarray] = []
    rng = np.random.default_rng(RANDOM_SEED + 200000 + generation + batch_index * 100003)
    parent_pair_offset = batch_index * WORKING_POPULATION_SIZE

    for child_index in range(candidate_count):
        child = make_unique_child_from_parents(
            ranked_parent_vectors,
            parent_pair_offset + child_index,
            rng,
            used_keys,
        )
        children.append(child)

    return np.asarray(children, dtype=float)


def evaluate_archive_generation_with_replacements(
    state: CalibrationState,
    parent_records: list[dict[str, Any]],
    generation: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    feasible_count = 0
    replacement_batch = 0
    max_evaluations = WORKING_POPULATION_SIZE * MAX_FEASIBILITY_EVALUATION_MULTIPLIER

    while feasible_count < WORKING_POPULATION_SIZE:
        remaining_budget = max_evaluations - len(records)
        remaining_replacement_budget = remaining_feasibility_replacement_budget(state)
        if remaining_budget <= 0 or (replacement_batch > 0 and remaining_replacement_budget <= 0):
            exit_for_insufficient_feasible_candidates(
                state=state,
                generation=generation,
                target_count=WORKING_POPULATION_SIZE,
                feasible_count=feasible_count,
                evaluated_count=len(records),
                maximum_evaluation_count=max_evaluations,
                phase="same-generation parent crossover replacements",
            )

        missing_count = WORKING_POPULATION_SIZE - feasible_count
        candidate_count = min(missing_count, remaining_budget)
        if replacement_batch > 0:
            candidate_count = reserve_feasibility_replacements(state, candidate_count)
        child_population = generate_archive_children(
            parent_records,
            state,
            generation,
            candidate_count=candidate_count,
            batch_index=replacement_batch,
        )
        label = "archive-child" if replacement_batch == 0 else "archive-feasibility-replacement"
        batch_records = evaluate_generation_vectors(state, child_population, generation, label)
        batch_feasible_count = sum(record.get("is_feasible", False) for record in batch_records)
        role = "archive_child" if replacement_batch == 0 else "archive_feasibility_replacement"
        for record in batch_records:
            record["generation_candidate_role"] = role
            record["feasibility_replacement_batch"] = replacement_batch

        records.extend(batch_records)
        feasible_count += batch_feasible_count
        save_removed_candidate_records(state)

        if batch_feasible_count < len(batch_records):
            log(
                f"Generation {generation}: feasibility batch {replacement_batch} retained "
                f"{batch_feasible_count}/{len(batch_records)} candidate(s); "
                f"{WORKING_POPULATION_SIZE - feasible_count} replacement slot(s) remain"
            )
        replacement_batch += 1

    return records


def save_best_candidate_media(best_record: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    media_workspace = run_dir / "best_media_workspace"
    if media_workspace.exists():
        shutil.rmtree(media_workspace)

    copy_source_workspace(media_workspace)
    patch_parameter_file(media_workspace, best_record["parameters"])
    run_result = run_example(media_workspace, save_media=True)
    return {
        "workspace": str(media_workspace),
        "returncode": run_result["returncode"],
        "runtime_seconds": round(run_result["runtime_seconds"], 4),
        "metrics_path": str(media_workspace / METRICS_FILE),
        "media_dir": str(media_workspace / ".images"),
    }


def best_record_from_population(state: CalibrationState, population: np.ndarray) -> dict[str, Any]:
    records = state.records_for_population(np.asarray(population, dtype=float))
    if not records:
        raise RuntimeError("Final population could not be matched to evaluated records.")
    missing_scores = [record["candidate_id"] for record in records if "final_topsis_score" not in record]
    if missing_scores:
        raise RuntimeError(
            "Final population records were not rescored with final TOPSIS values: "
            + ", ".join(missing_scores)
        )
    ranked = sorted(
        records,
        key=lambda record: float(record.get("final_topsis_score", MIN_TOPSIS_FITNESS)),
        reverse=True,
    )
    best = dict(ranked[0])
    best["final_pool_topsis_score"] = round(float(best["final_topsis_score"]), 9)
    return best


def plot_reports(generation_summaries: list[dict[str, Any]], all_records: list[dict[str, Any]], run_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        log(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    generations = [summary["generation"] for summary in generation_summaries]
    record_by_id = {record["candidate_id"]: record for record in all_records}

    def save_line_plot(
        values: list[float],
        labels: list[str],
        title: str,
        ylabel: str,
        output_name: str,
        label_rotation: int = 15,
        label_y_offset: int = 5,
        annotate_changes_only: bool = False,
        shorten_candidate_labels: bool = False,
    ) -> None:
        plt.figure(figsize=(11, 5))
        plt.plot(generations, values, marker="o")
        previous_label = None
        for x_value, y_value, label in zip(generations, values, labels):
            if annotate_changes_only and label == previous_label:
                continue
            previous_label = label
            display_label = label
            if shorten_candidate_labels and label.startswith("ga_c"):
                display_label = f"C{int(label.split('ga_c', 1)[1])}"
            plt.annotate(
                display_label,
                (x_value, y_value),
                textcoords="offset points",
                xytext=(4, label_y_offset),
                fontsize=8,
                rotation=label_rotation,
            )
        plt.title(title)
        plt.xlabel("Generation")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / output_name, dpi=150)
        plt.close()

    generation_best_ids = [summary.get("best_candidate_id") or f"gen_{summary['generation']}" for summary in generation_summaries]
    best_so_far_ids = [
        summary.get("best_so_far_candidate_id") or summary.get("best_candidate_id") or f"gen_{summary['generation']}"
        for summary in generation_summaries
    ]

    def final_score(candidate_id: str) -> float:
        record = record_by_id.get(candidate_id)
        if record is None:
            return MIN_TOPSIS_FITNESS
        return float(record.get("final_topsis_score", record.get("current_topsis_score", MIN_TOPSIS_FITNESS)))

    save_line_plot(
        [final_score(candidate_id) for candidate_id in generation_best_ids],
        generation_best_ids,
        "Generation-Best Candidate Final TOPSIS Score",
        "Final TOPSIS score, higher is better",
        "generation_best_final_topsis_score_by_generation.png",
    )

    save_line_plot(
        [final_score(candidate_id) for candidate_id in best_so_far_ids],
        best_so_far_ids,
        "Best TOPSIS score by generation",
        "Final TOPSIS score, higher is better",
        "best_so_far_final_topsis_score_by_generation.png",
        label_rotation=0,
        label_y_offset=-14,
        annotate_changes_only=True,
        shorten_candidate_labels=True,
    )

    save_line_plot(
        [float(summary["average_topsis_score"]) for summary in generation_summaries],
        generation_best_ids,
        "Average Cumulative TOPSIS Score by Generation",
        "Average TOPSIS score at generation",
        "average_topsis_score_by_generation.png",
    )

    for metric in TOPSIS_METRICS:
        values = []
        for candidate_id in generation_best_ids:
            record = record_by_id.get(candidate_id, {})
            values.append(float(record.get("topsis_metrics", {}).get(metric, default_for_metric(metric))))
        save_line_plot(
            values,
            generation_best_ids,
            f"Generation-Best Candidate {metric}",
            metric,
            f"generation_best_candidate_{metric}_by_generation.png",
        )


def write_report(run_dir: Path, generation_summaries: list[dict[str, Any]], best_record: dict[str, Any], media_result: dict[str, Any] | None) -> None:
    total_removed = sum(int(summary.get("removed_candidate_count", 0)) for summary in generation_summaries)
    lines = [
        "# Example 10 Archive Genetic Algorithm Calibration",
        "",
        f"- Selection library: `pymcdm.methods.TOPSIS`",
        f"- Selection rule: after every generation, rescore feasible candidates and choose up to the top {WORKING_POPULATION_SIZE} as parents",
        f"- Feasibility rule: `{FEASIBILITY_METRIC} <= {MAX_AVERAGE_OVERLAP_DURATION_SECONDS}`",
        f"- Feasibility replacement: refill removed child slots using the same generation parents; generation 1 uses unique random replacements",
        f"- Maximum evaluations while refilling: {MAX_FEASIBILITY_EVALUATION_MULTIPLIER}x the target population",
        f"- Maximum feasibility replacements across the complete calibration: {MAX_FEASIBILITY_REPLACEMENTS_TOTAL}",
        f"- Feasibility replacements evaluated: {active_state().feasibility_replacements_evaluated}",
        f"- Removed infeasible candidates: {total_removed}",
        f"- Removed-candidate records: `{run_dir / 'removed_candidates.json'}`",
        f"- TOPSIS ideal best/worst: derived from feasible candidates evaluated so far, not preset",
        f"- TOPSIS score: higher is better",
        f"- Initial candidates: {INITIAL_POPULATION_SIZE}",
        f"- Candidates per generation: {WORKING_POPULATION_SIZE}",
        f"- Archive parent pool size: {WORKING_POPULATION_SIZE}",
        f"- Total generations: {GENERATIONS}",
        f"- Crossover probability: {CROSSOVER_PROBABILITY}",
        f"- Mutation probability: {MUTATION_PROBABILITY}",
        f"- Parameter mutation sizes: `{json.dumps(PARAMETER_MUTATION_SIZES)}`",
        f"- Random seed: {RANDOM_SEED}",
        f"- Per-candidate simulation repeats: {CALIBRATION_REPEAT_COUNT}",
        f"- Per-candidate repeat seeds: {CALIBRATION_REPEAT_SEEDS[:CALIBRATION_REPEAT_COUNT]}",
        "",
        "## TOPSIS Criteria",
        "",
        "| Metric | Direction | Weight |",
        "| --- | --- | ---: |",
    ]
    for metric in TOPSIS_METRICS:
        lines.append(f"| {metric} | {METRIC_DIRECTIONS[metric]} | {TOPSIS_WEIGHTS[metric]} |")

    lines.extend(["", "## Generation Best Candidates", ""])
    lines.append("| Generation | Candidate | TOPSIS score at time | Best so far | Goal reach rate | Avg completion time | Avg overlap duration |")
    lines.append("| ---: | --- | ---: | --- | ---: | ---: | ---: |")
    for summary in generation_summaries:
        metrics = summary.get("best_candidate_topsis_metrics") or {}
        lines.append(
            f"| {summary['generation']} | {summary['best_candidate_id']} | "
            f"{summary['best_candidate_topsis_score_at_time']} | "
            f"{summary['best_so_far_candidate_id']} | "
            f"{metrics.get('goal_reach_rate')} | "
            f"{metrics.get('average_completion_time')} | "
            f"{metrics.get('average_overlap_duration_seconds')} |"
        )

    lines.extend(["", "## Final Selected Candidate", ""])
    lines.append(f"- Candidate: {best_record['candidate_id']}")
    lines.append(f"- Source generation: {best_record['generation']}")
    lines.append(f"- Final pool TOPSIS score: {best_record.get('final_pool_topsis_score')}")
    lines.append(f"- Final TOPSIS score: {best_record.get('final_topsis_score')}")
    lines.append(f"- Parameters: `{json.dumps(best_record['parameters'])}`")
    lines.append(f"- TOPSIS metrics: `{json.dumps(best_record['topsis_metrics'])}`")
    lines.append(f"- Observed metrics: `{json.dumps(best_record['observed_metrics'])}`")
    if media_result:
        lines.append(f"- Best media workspace: `{media_result['workspace']}`")
        lines.append(f"- Best media directory: `{media_result['media_dir']}`")

    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Archive GA calibration for Test-new/example10.py")
    parser.add_argument("--save_best_media", action="store_true")
    return parser.parse_args()


def main() -> int:
    global _ACTIVE_STATE

    args = parse_args()

    if TOPSIS is None:
        raise SystemExit(
            "pymcdm is not installed in this environment. Install it with: "
            "python -m pip install pymcdm"
        )

    run_dir = SCRIPT_DIR / "calibration" / f"archive_ga_example10_{now_id()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "workspaces").mkdir(parents=True, exist_ok=True)

    lower, upper = parameter_bounds()
    state = CalibrationState(run_dir)
    _ACTIVE_STATE = state

    log(f"Archive GA calibration output: {run_dir}", detail=True)
    log(f"Using parameters: {', '.join(PARAMETER_SPECS)}")
    log(f"Parameter lower bounds: {json.dumps(dict(zip(parameter_names(), lower.tolist())))}", detail=True)
    log(f"Parameter upper bounds: {json.dumps(dict(zip(parameter_names(), upper.tolist())))}", detail=True)
    log(f"TOPSIS metrics: {json.dumps(TOPSIS_METRICS)}", detail=True)
    log(f"TOPSIS weights: {json.dumps(TOPSIS_WEIGHTS)}", detail=True)
    log(f"Parameter mutation sizes: {json.dumps(PARAMETER_MUTATION_SIZES)}", detail=True)
    log(f"Archive parent pool size: {WORKING_POPULATION_SIZE}", detail=True)
    log(
        f"Feasibility: {FEASIBILITY_METRIC} <= "
        f"{MAX_AVERAGE_OVERLAP_DURATION_SECONDS}; infeasible candidates are removed from selection"
    )
    log("Selection: cumulative pymcdm TOPSIS over feasible candidates only; higher is better", detail=True)

    initial_population = generate_initial_population()
    evaluate_initial_population(state, initial_population, generation=1)
    parent_records = select_archive_parent_records(state)
    log_generation(
        1,
        f"Generation 1: selected {len(parent_records)} feasible archive candidate(s) "
        "as parents for generation 2"
    )

    for generation in range(2, GENERATIONS + 1):
        log(
            f"Generation {generation}: creating {WORKING_POPULATION_SIZE} child candidate(s) "
            f"from {len(parent_records)} feasible cumulative TOPSIS parent(s)",
            detail=True,
        )
        records = evaluate_archive_generation_with_replacements(state, parent_records, generation)
        rescore_records_with_topsis(state.all_records)
        summary = save_generation_outputs(records, state.run_dir, generation)
        parent_records = select_archive_parent_records(state)
        summary["archive_parent_candidate_ids_for_next_generation"] = [
            record["candidate_id"] for record in parent_records
        ]
        generation_dir = state.run_dir / f"generation_{generation:03d}"
        (generation_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        state.generation_summaries.append(summary)
        log_generation(
            generation,
            f"Generation {generation}: best generation TOPSIS "
            f"{summary['best_candidate_topsis_score_at_time']} from {summary['best_candidate_id']}; "
            f"removed {summary['removed_candidate_count']} infeasible candidate(s); "
            f"selected {len(parent_records)} feasible archive parent(s) for next generation"
        )

    rescore_records_with_topsis(state.all_records, score_key="final_topsis_score")
    final_parent_population = parent_records_to_population(parent_records)
    best_record = best_record_from_population(state, final_parent_population)

    media_result = None
    if args.save_best_media:
        log("Rerunning final GA-best candidate with media enabled")
        media_result = save_best_candidate_media(best_record, run_dir)

    (run_dir / "all_records.json").write_text(json.dumps(state.all_records, indent=2), encoding="utf-8")
    (run_dir / "removed_candidates.json").write_text(
        json.dumps(state.removed_records, indent=2),
        encoding="utf-8",
    )
    (run_dir / "generation_summaries.json").write_text(json.dumps(state.generation_summaries, indent=2), encoding="utf-8")
    (run_dir / "best_calibration.json").write_text(
        json.dumps(
            {
                "best_candidate": best_record,
                "media_result": media_result,
                "settings": {
                    "ga_library": "custom archive GA loop with library TOPSIS selection",
                    "selection": "top-N cumulative pymcdm TOPSIS archive parent selection",
                    "topsis_reference": "ideal best/worst values are derived by pymcdm from feasible evaluated candidates, not preset",
                    "feasibility_metric": FEASIBILITY_METRIC,
                    "feasibility_operator": "<=",
                    "feasibility_threshold": MAX_AVERAGE_OVERLAP_DURATION_SECONDS,
                    "feasibility_replacement": "same-generation parent crossover and mutation",
                    "max_feasibility_evaluation_multiplier": MAX_FEASIBILITY_EVALUATION_MULTIPLIER,
                    "max_feasibility_replacements_total": MAX_FEASIBILITY_REPLACEMENTS_TOTAL,
                    "feasibility_replacements_evaluated": state.feasibility_replacements_evaluated,
                    "removed_candidate_count": len(state.removed_records),
                    "initial_population_size": INITIAL_POPULATION_SIZE,
                    "candidates_per_generation": WORKING_POPULATION_SIZE,
                    "archive_parent_pool_size": WORKING_POPULATION_SIZE,
                    "total_generations": GENERATIONS,
                    "crossover_probability": CROSSOVER_PROBABILITY,
                    "mutation_probability": MUTATION_PROBABILITY,
                    "parameter_mutation_sizes": PARAMETER_MUTATION_SIZES,
                    "random_seed": RANDOM_SEED,
                    "calibration_repeat_count": CALIBRATION_REPEAT_COUNT,
                    "calibration_repeat_seeds": CALIBRATION_REPEAT_SEEDS[:CALIBRATION_REPEAT_COUNT],
                    "parameter_specs": PARAMETER_SPECS,
                    "topsis_metrics": TOPSIS_METRICS,
                    "topsis_weights": TOPSIS_WEIGHTS,
                    "metric_directions": METRIC_DIRECTIONS,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_reports(state.generation_summaries, state.all_records, run_dir)
    write_report(run_dir, state.generation_summaries, best_record, media_result)

    _ACTIVE_STATE = None
    log("Archive GA calibration complete")
    log(f"Best candidate: {best_record['candidate_id']}")
    log(f"Parameters: {json.dumps(best_record['parameters'])}")
    log(f"Final TOPSIS score: {best_record.get('final_topsis_score')}")
    log(f"TOPSIS metrics: {json.dumps(best_record['topsis_metrics'])}", detail=True)
    log(f"Observed metrics: {json.dumps(best_record['observed_metrics'])}", detail=True)
    log(f"Removed infeasible candidates: {len(state.removed_records)}")
    log(f"Removed candidate records: {run_dir / 'removed_candidates.json'}", detail=True)
    log(f"Report: {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

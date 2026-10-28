from __future__ import annotations

import argparse
import ast
import concurrent.futures
import json
import math
import os
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from archive_ga_calibrator import ArchiveGACalibrator
from calibration_candidate_runner import CalibrationCandidateRunner
from context_compactor import ContextCompactor
from parameter_space import CALIBRATION_MANIFEST_NAME, validate_manifest_inheritance
from evaluator import Evaluator
from history_manager import HistoryManager
from llm_client import LLMClient
from openai_key import resolve_openai_api_key
from patch_manager import PatchManager
from simulation_runner import SimulationRunner
from video_analyzer import VideoAnalyzer


PROJECT_DIR = Path(__file__).resolve().parent
MAX_CHILDREN_PER_ITERATION = 30
MAX_PARALLEL_MEDIA_RUNS = 10
TERMINAL_VERBOSITY = "concise"
DOMINATION_METRICS = {
    "goal_reach_rate": "higher",
    "average_overlap_duration_seconds": "lower",
    "standard_cyclomatic_complexity": "lower",
    "halstead_effort": "lower",
    "average_completion_time": "lower",
}


def checkpoint(message: str, detail: bool = False) -> None:
    if detail and TERMINAL_VERBOSITY != "detailed":
        return
    terminal_print(message)


def terminal_print(message: Any = "") -> None:
    """Print controller output with a timestamp on every emitted line."""
    text = str(message)
    lines = text.splitlines() or [""]
    for line in lines:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {line}", flush=True)


def resolve_parallel_workers(configured: Any, job_count: int) -> int:
    if job_count <= 0:
        return 0
    if str(configured).strip().lower() == "all":
        return max(1, min(os.cpu_count() or 1, job_count))
    return max(1, min(int(configured), job_count))


def divided_cpu_environment(worker_count: int) -> dict[str, str]:
    cores_per_worker = max(1, (os.cpu_count() or 1) // max(1, worker_count))
    thread_count = str(cores_per_worker)
    return {
        "OMP_NUM_THREADS": thread_count,
        "MKL_NUM_THREADS": thread_count,
        "OPENBLAS_NUM_THREADS": thread_count,
        "NUMEXPR_NUM_THREADS": thread_count,
        "XLA_FLAGS": (
            "--xla_cpu_multi_thread_eigen=true "
            f"intra_op_parallelism_threads={thread_count}"
        ),
    }


def timestamped_input(message: str) -> str:
    timestamp = datetime.now().strftime("%H:%M:%S")
    return input(f"[{timestamp}] {message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated SFM refinement agent.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    parser.add_argument("--scenario", help="Scenario alias, e.g. scenario1 maps to example1.")
    parser.add_argument("--all_scenarios", action="store_true", help="Run all configured examples.")
    parser.add_argument("--example", type=int, action="append", help="Example number to run; repeatable.")
    parser.add_argument("--example10", action="store_true", help="Shortcut for --example 10.")
    parser.add_argument("--all_examples", action="store_true", help="Run all discovered/configured examples.")
    parser.add_argument("--max_iterations", type=int, help="Override config max_iterations.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue refinement from history/pool_state.json when available; "
            "otherwise use the latest accepted version."
        ),
    )
    parser.add_argument(
        "--start_iteration",
        type=int,
        help=(
            "Iteration number to start from. With --resume, defaults to the "
            "saved pool iteration plus one when pool_state.json exists."
        ),
    )
    parser.add_argument("--baseline_only", action="store_true", help="Run/evaluate baseline and stop.")
    parser.add_argument(
        "--skip_base_calibration",
        "--skip-base-calibration",
        action="store_true",
        help=(
            "Do not GA-calibrate the base model before refinement; use the "
            "current/default base parameters as already calibrated. Child "
            "candidates are still calibrated normally."
        ),
    )
    parser.add_argument("--calibration", action="store_true", help="Run example-10 archive GA calibration instead of LLM refinement.")
    parser.add_argument("--run_calibration_candidate", type=int, help="Run one candidate from a ranked GA candidate pool by 1-based number.")
    parser.add_argument("--calibration_run_dir", help="Calibration run directory to read; defaults to the latest run with candidate_pool.json.")
    parser.add_argument("--candidate_no_media", action="store_true", help="Replay selected calibration candidate without PNG/MP4 output.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        return load_simple_yaml(text)


def load_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for this config file when PyYAML is unavailable."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        if not value:
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = parse_scalar(value)
    return root


def parse_scalar(value: str) -> Any:
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.lower() in {"null", "none"}:
        return None
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value.strip('"').strip("'")


def determine_examples(args: argparse.Namespace, config: dict[str, Any]) -> list[int] | None:
    if args.example10:
        return [10]
    if args.example:
        return args.example
    if args.scenario:
        lowered = args.scenario.lower()
        if lowered.startswith("scenario"):
            return [int(lowered.replace("scenario", ""))]
        if lowered.startswith("example"):
            return [int(lowered.replace("example", ""))]
        raise ValueError(f"Unsupported scenario name: {args.scenario}")
    if args.all_scenarios or args.all_examples:
        return list(config.get("example_numbers", []))
    if args.run_calibration_candidate:
        return None
    return list(config.get("example_numbers", []))


def latest_history_iteration(history: HistoryManager) -> int:
    iterations = [
        int(item.get("iteration", 0))
        for item in history.load()
        if item.get("iteration") is not None
    ]
    return max(iterations, default=0)


def iteration_bounds(
    args: argparse.Namespace,
    config: dict[str, Any],
    history: HistoryManager,
    pool_state: dict[str, Any] | None = None,
) -> tuple[int, int]:
    max_iterations = int(config.get("max_iterations", 5))
    if args.start_iteration is not None:
        start_iteration = args.start_iteration
    elif args.resume and pool_state:
        start_iteration = int(pool_state.get("last_completed_iteration", 0) or 0) + 1
    elif args.resume:
        start_iteration = latest_history_iteration(history) + 1
    else:
        start_iteration = 1
    if start_iteration < 1:
        raise ValueError("--start_iteration must be 1 or greater.")
    return start_iteration, max_iterations


def resume_command(
    args: argparse.Namespace,
    next_iteration: int,
    target_max_iterations: int | None = None,
) -> str:
    parts = [quote_cli_arg(sys.executable), "main.py"]
    if args.config != "config.yaml":
        parts.extend(["--config", quote_cli_arg(args.config)])
    if args.example10:
        parts.append("--example10")
    elif args.example:
        for example in args.example:
            parts.extend(["--example", str(example)])
    elif args.scenario:
        parts.extend(["--scenario", quote_cli_arg(args.scenario)])
    elif args.all_scenarios:
        parts.append("--all_scenarios")
    elif args.all_examples:
        parts.append("--all_examples")
    if getattr(args, "skip_base_calibration", False):
        parts.append("--skip_base_calibration")
    parts.extend(["--resume", "--start_iteration", str(next_iteration)])
    if target_max_iterations is not None:
        parts.extend(["--max_iterations", str(target_max_iterations)])
    elif args.max_iterations is not None:
        parts.extend(["--max_iterations", str(args.max_iterations)])
    return " ".join(parts)


def quote_cli_arg(value: str) -> str:
    if any(char.isspace() for char in value):
        return f'"{value}"'
    return value


def print_resume_hint(
    args: argparse.Namespace,
    next_iteration: int,
    target_max_iterations: int | None = None,
) -> None:
    terminal_print("Resume command:")
    terminal_print(resume_command(args, next_iteration, target_max_iterations))


def read_code_excerpt(workspace: Path, config: dict[str, Any], max_chars: int | None) -> str:
    paths = [
        workspace / CALIBRATION_MANIFEST_NAME,
        workspace / config.get("model_file_name", "jsfm/sfm.py"),
    ]
    paths.extend(sorted(workspace.glob(config.get("example_glob", "example*.py"))))
    for pattern in config.get("additional_tracked_globs", []):
        paths.extend(sorted(workspace.glob(pattern)))
    chunks: list[str] = []
    unlimited = max_chars is None or max_chars <= 0
    remaining = max_chars or 0
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists() or (not unlimited and remaining <= 0):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunk = f"\n# File: {path.relative_to(workspace).as_posix()}\n{text}"
        if not unlimited and len(chunk) > remaining:
            chunks.append(
                f"\n# File omitted: {path.relative_to(workspace).as_posix()} "
                "would exceed max_code_chars_for_llm. Set max_code_chars_for_llm "
                "to 0 to send complete tracked files."
            )
            continue
        chunks.append(chunk)
        if not unlimited:
            remaining -= len(chunk)
    return "\n".join(chunks)


def select_candidate(diagnosis: dict[str, Any]) -> dict[str, Any]:
    recommended = diagnosis.get("recommended_candidate_id")
    for candidate in diagnosis.get("candidates", []):
        if candidate.get("id") == recommended:
            return candidate
    return max(diagnosis["candidates"], key=lambda item: item.get("priority_score", 0.0))


def select_child_candidates(diagnosis: dict[str, Any], count: int) -> list[dict[str, Any]]:
    candidates = list(diagnosis.get("candidates", []))
    recommended = diagnosis.get("recommended_candidate_id")
    candidates.sort(
        key=lambda item: (
            0 if item.get("id") == recommended else 1,
            -float(item.get("priority_score", 0.0) or 0.0),
            int(item.get("id", 9999) or 9999),
        )
    )
    return candidates[:count]


def allocate_child_budget(
    pool: list[dict[str, Any]],
    max_children: int,
    iteration: int,
    seed: int = 42,
) -> list[tuple[dict[str, Any], int]]:
    active = [candidate for candidate in pool if candidate.get("status") == "active"]
    if not active or max_children <= 0:
        return []

    ranked = rank_pool_candidates_with_topsis(active)
    if len(ranked) >= max_children:
        selected = ranked[:max_children]
    else:
        selected = ranked

    allocation = {candidate["candidate_id"]: 1 for candidate in selected}
    remaining = max_children - len(selected)
    rng = random.Random(seed + iteration)
    while remaining > 0 and selected:
        candidate = rng.choice(selected)
        allocation[candidate["candidate_id"]] += 1
        remaining -= 1
    return [(candidate, allocation[candidate["candidate_id"]]) for candidate in selected]


def rank_pool_candidates_with_topsis(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_rows: list[tuple[dict[str, Any], list[float]]] = []
    invalid: list[dict[str, Any]] = []
    for candidate in candidates:
        row: list[float] = []
        for metric in DOMINATION_METRICS:
            value = metric_value(candidate, metric)
            if value is None or not math.isfinite(value):
                invalid.append(candidate)
                break
            row.append(value)
        else:
            valid_rows.append((candidate, row))

    if not valid_rows:
        for index, candidate in enumerate(candidates, start=1):
            candidate["pool_topsis_rank"] = index
            candidate["pool_topsis_score"] = None
            candidate["pool_topsis_selection_rule"] = "Fallback order because TOPSIS metrics were incomplete."
        return list(candidates)

    column_count = len(DOMINATION_METRICS)
    denominators: list[float] = []
    for column_index in range(column_count):
        denominator = math.sqrt(sum(row[column_index] ** 2 for _, row in valid_rows))
        denominators.append(denominator if denominator > 0.0 else 1.0)

    normalized_rows = [
        [row[column_index] / denominators[column_index] for column_index in range(column_count)]
        for _, row in valid_rows
    ]

    ideal: list[float] = []
    worst: list[float] = []
    directions = list(DOMINATION_METRICS.values())
    for column_index, direction in enumerate(directions):
        column = [row[column_index] for row in normalized_rows]
        if direction == "higher":
            ideal.append(max(column))
            worst.append(min(column))
        else:
            ideal.append(min(column))
            worst.append(max(column))

    scored: list[tuple[dict[str, Any], float]] = []
    for (candidate, _), normalized in zip(valid_rows, normalized_rows):
        distance_to_ideal = math.sqrt(
            sum((normalized[index] - ideal[index]) ** 2 for index in range(column_count))
        )
        distance_to_worst = math.sqrt(
            sum((normalized[index] - worst[index]) ** 2 for index in range(column_count))
        )
        denominator = distance_to_ideal + distance_to_worst
        score = distance_to_worst / denominator if denominator else 0.0
        candidate["pool_topsis_score"] = round(score, 6)
        candidate["pool_topsis_distance_to_ideal"] = round(distance_to_ideal, 6)
        candidate["pool_topsis_distance_to_worst"] = round(distance_to_worst, 6)
        candidate["pool_topsis_selection_rule"] = "TOPSIS without user weights over active pool metrics."
        scored.append((candidate, score))

    ranked = [
        candidate
        for candidate, _ in sorted(
            scored,
            key=lambda item: (-item[1], str(item[0].get("candidate_id", ""))),
        )
    ]
    for index, candidate in enumerate(ranked, start=1):
        candidate["pool_topsis_rank"] = index

    for candidate in invalid:
        candidate["pool_topsis_rank"] = None
        candidate["pool_topsis_score"] = None
        candidate["pool_topsis_selection_rule"] = "Not ranked because one or more TOPSIS metrics were missing."
    return ranked + invalid


def pool_candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": candidate.get("candidate_id"),
        "parent_id": candidate.get("parent_id"),
        "iteration_created": candidate.get("iteration_created"),
        "has_produced_child": candidate.get("has_produced_child"),
        "produced_child_count": candidate.get("produced_child_count", 0),
        "last_produced_iteration": candidate.get("last_produced_iteration"),
        "status": candidate.get("status"),
        "dominated_by": candidate.get("dominated_by"),
        "dominated_order": candidate.get("dominated_order"),
        "autocred_plausibility": candidate.get("autocred_plausibility"),
        "replacement_attempt": candidate.get("replacement_attempt"),
        "replaces_candidate_id": candidate.get("replaces_candidate_id"),
        "base_calibration_skipped": candidate.get("base_calibration_skipped"),
        "pool_topsis_rank": candidate.get("pool_topsis_rank"),
        "pool_topsis_score": candidate.get("pool_topsis_score"),
        "metrics": {
            key: metric_value(candidate, key)
            for key in DOMINATION_METRICS
            if metric_value(candidate, key) is not None
        },
        "workspace": candidate.get("workspace"),
    }


def pool_summary(pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [pool_candidate_summary(candidate) for candidate in pool]


def _short_history_text(value: Any, limit: int = 6000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "...[truncated for diagnosis context]"


def _compact_autocred_for_diagnosis(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    compact = {
        key: value.get(key)
        for key in ("status", "score", "plausibility_score", "threshold", "plausibility_threshold")
        if value.get(key) is not None
    }
    feedback = value.get("feedback_summary") or value.get("reason")
    if feedback:
        compact["feedback_summary"] = _short_history_text(feedback, 1200)
    return compact or None


def _compact_candidate_history_for_diagnosis(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {"value": _short_history_text(candidate, 1000)}
    compact = {
        key: candidate.get(key)
        for key in (
            "candidate_id",
            "parent_id",
            "iteration_created",
            "status",
            "dominated_by",
            "dominated_order",
            "replacement_attempt",
            "replaces_candidate_id",
            "pool_topsis_rank",
            "pool_topsis_score",
            "metrics",
        )
        if candidate.get(key) is not None
    }
    autocred = _compact_autocred_for_diagnosis(candidate.get("autocred_plausibility"))
    if autocred:
        compact["autocred_plausibility"] = autocred
    return compact


def compact_history_for_diagnosis(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep scientific context while removing repeated pool bookkeeping."""
    compact_history: list[dict[str, Any]] = []
    for entry in history:
        compact: dict[str, Any] = {
            key: entry.get(key)
            for key in ("mode", "iteration", "scenario", "examples", "decision", "notes")
            if entry.get(key) is not None
        }
        diagnosis = entry.get("llm_diagnosis")
        if diagnosis:
            compact["previous_diagnosis"] = _short_history_text(diagnosis)

        children = entry.get("children_created") or []
        if children:
            compact["children_created"] = [
                _compact_candidate_history_for_diagnosis(candidate)
                for candidate in children
            ]

        archived = entry.get("archived_children") or []
        if archived:
            compact["archived_children"] = [
                {
                    key: item.get(key)
                    for key in ("candidate_id", "status")
                    if item.get(key) is not None
                }
                for item in archived
                if isinstance(item, dict)
            ]

        mappings = entry.get("literature_mappings") or []
        if mappings:
            compact["literature_mappings"] = [
                {
                    key: _short_history_text(item.get(key), 1200)
                    if key in {"diagnosis_or_cause", "relevant_concept", "how_it_maps_to_this_code_change", "source"}
                    else item.get(key)
                    for key in (
                        "kind",
                        "parent_candidate_id",
                        "candidate_id",
                        "source",
                        "source_type",
                        "diagnosis_or_cause",
                        "relevant_concept",
                        "how_it_maps_to_this_code_change",
                        "confidence",
                        "confidence_reason",
                    )
                    if item.get(key) is not None
                }
                for item in mappings
                if isinstance(item, dict)
            ]
        compact_history.append(compact)
    return compact_history


def diagnosis_history_context(history: HistoryManager, config: dict[str, Any]) -> list[dict[str, Any]]:
    return ContextCompactor(config, history.history_dir).build(history.load())


def collect_literature_mappings(
    diagnoses: list[dict[str, Any]],
    children: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    for diagnosis in diagnoses:
        parent_id = diagnosis.get("parent_candidate_id")
        for item in diagnosis.get("diagnosis_literature_mapping", []) or []:
            mappings.append(
                {
                    "kind": "diagnosis",
                    "parent_candidate_id": parent_id,
                    "diagnosis_or_cause": item.get("diagnosis_or_cause"),
                    "source": item.get("source"),
                    "source_type": item.get("source_type"),
                    "relevant_concept": item.get("how_it_supports_the_diagnosis"),
                    "confidence": item.get("confidence"),
                    "confidence_reason": item.get("confidence_reason"),
                }
            )
    for child in children:
        llm_candidate = child.get("llm_candidate", {}) or {}
        for item in llm_candidate.get("literature_basis", []) or []:
            mappings.append(
                {
                    "kind": "refinement",
                    "candidate_id": child.get("candidate_id"),
                    "parent_candidate_id": child.get("parent_id"),
                    "candidate_name": llm_candidate.get("name"),
                    "source": item.get("source"),
                    "source_type": item.get("source_type"),
                    "relevant_concept": item.get("relevant_concept"),
                    "how_it_maps_to_this_code_change": item.get("how_it_maps_to_this_code_change"),
                    "confidence": item.get("confidence"),
                    "confidence_reason": item.get("confidence_reason"),
                }
            )
    return mappings


def make_pool_state(
    candidate_pool: list[dict[str, Any]],
    last_completed_iteration: int,
    examples: list[int] | None,
    baseline_lines: int,
    domination_sequence: int,
) -> dict[str, Any]:
    return {
        "version": 1,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "last_completed_iteration": last_completed_iteration,
        "examples": examples,
        "baseline_lines": baseline_lines,
        "domination_sequence": domination_sequence,
        "candidate_pool": candidate_pool,
    }


def restore_pool_state(state: dict[str, Any]) -> tuple[list[dict[str, Any]], int, list[int] | None, int | None]:
    pool = list(state.get("candidate_pool", []))
    for candidate in pool:
        workspace = candidate.get("workspace")
        if workspace and not Path(workspace).exists() and candidate.get("status") == "active":
            candidate["status"] = "failed"
            candidate["failure_reason"] = "Workspace was missing when pool resume was attempted."
    domination_sequence = int(state.get("domination_sequence", 0) or 0)
    examples = state.get("examples")
    if examples is not None:
        examples = [int(value) for value in examples]
    baseline_lines = state.get("baseline_lines")
    if baseline_lines is not None:
        baseline_lines = int(baseline_lines)
    return pool, domination_sequence, examples, baseline_lines


def first_active_workspace(candidate_pool: list[dict[str, Any]]) -> Path | None:
    for candidate in candidate_pool:
        if candidate.get("status") != "active":
            continue
        workspace = candidate.get("workspace")
        if workspace and Path(workspace).exists():
            return Path(workspace)
    return None


def metric_value(candidate: dict[str, Any], metric: str) -> float | None:
    metrics = candidate.get("metrics", {})
    value = to_float_or_none(metrics.get(metric))
    if value is not None and not (metric == "goal_reach_rate" and value == 0.0):
        return value
    scenario_values = [
        to_float_or_none(item.get(metric))
        for item in metrics.get("scenario_metrics", []) or []
        if isinstance(item, dict)
    ]
    scenario_values = [item for item in scenario_values if item is not None]
    if not scenario_values:
        return value
    if metric == "goal_reach_rate":
        total_weight = 0.0
        weighted = 0.0
        for item in metrics.get("scenario_metrics", []) or []:
            if not isinstance(item, dict):
                continue
            rate = to_float_or_none(item.get(metric))
            if rate is None:
                continue
            weight = to_float_or_none(item.get("total_pedestrians")) or 1.0
            weighted += rate * weight
            total_weight += weight
        return weighted / total_weight if total_weight else scenario_values[0]
    return scenario_values[0] if value is None else value


def strictly_better(a_value: float, b_value: float, direction: str) -> bool:
    if direction == "higher":
        return a_value > b_value
    return a_value < b_value


def hard_dominates(candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> bool:
    for metric, direction in DOMINATION_METRICS.items():
        a_value = metric_value(candidate_a, metric)
        b_value = metric_value(candidate_b, metric)
        if a_value is None or b_value is None:
            return False
        if not strictly_better(a_value, b_value, direction):
            return False
    return True


def apply_hard_domination(
    pool: list[dict[str, Any]],
    domination_sequence: int,
) -> tuple[list[dict[str, Any]], int]:
    active = [candidate for candidate in pool if candidate.get("status") == "active"]
    for candidate in active:
        dominator = next(
            (
                other
                for other in active
                if other is not candidate and hard_dominates(other, candidate)
            ),
            None,
        )
        if dominator:
            candidate["status"] = "dominated"
            candidate["dominated_by"] = dominator.get("candidate_id")
            domination_sequence += 1
            candidate["dominated_order"] = domination_sequence
            candidate["domination_rule"] = "Removed only because another candidate is strictly better in all domination metrics."
    return pool, domination_sequence


def child_candidate_id(
    iteration: int,
    parent: dict[str, Any],
    index: int,
    attempt: int = 1,
) -> str:
    parent_id = str(parent.get("candidate_id", "parent")).replace(" ", "_")
    candidate_id = f"iter{iteration:03d}_{parent_id}_child{index:02d}"
    if attempt > 1:
        candidate_id += f"_try{attempt:02d}"
    return candidate_id


def safe_path_token(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def promote_calibrated_workspace(
    child: dict[str, Any],
    best_workspace: Path,
    config: dict[str, Any],
) -> Path:
    calibration_config = config.get("calibration", {})
    permanent_root = PROJECT_DIR / calibration_config.get(
        "promoted_workspace_dir",
        "model_store",
    )
    permanent_root.mkdir(parents=True, exist_ok=True)
    token = safe_path_token(str(child.get("candidate_id", "candidate")))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = permanent_root / f"{token}_{timestamp}"
    shutil.copytree(
        best_workspace,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", ".mypy_cache"),
    )
    return destination


def cleanup_calibration_run(result: dict[str, Any], config: dict[str, Any]) -> None:
    calibration_config = config.get("calibration", {})
    if not calibration_config.get("cleanup_after_success", False):
        return
    run_dir_value = result.get("run_dir")
    if not run_dir_value:
        return
    run_dir = Path(run_dir_value).resolve()
    calibration_root = (PROJECT_DIR / "calibration").resolve()
    if calibration_root not in run_dir.parents:
        raise RuntimeError(f"Refusing to delete calibration run outside {calibration_root}: {run_dir}")
    if run_dir.exists():
        shutil.rmtree(run_dir)


def cleanup_disposable_workspace(workspace_value: str | None, config: dict[str, Any]) -> None:
    calibration_config = config.get("calibration", {})
    if not calibration_config.get("cleanup_scratch_workspaces", False):
        return
    if not workspace_value:
        return
    workspace = Path(workspace_value).resolve()
    if not workspace.exists() or not workspace.is_dir():
        return
    allowed_roots = [
        (PROJECT_DIR / "working_model").resolve(),
        (PROJECT_DIR / calibration_config.get("promoted_workspace_dir", "model_store")).resolve(),
    ]
    if any(root == workspace for root in allowed_roots):
        raise RuntimeError(f"Refusing to delete managed workspace root: {workspace}")
    if not any(root in workspace.parents for root in allowed_roots):
        raise RuntimeError(f"Refusing to delete workspace outside managed roots: {workspace}")
    shutil.rmtree(workspace)


def output_dirs_for_candidate(candidate: dict[str, Any]) -> list[Path]:
    dirs: list[Path] = []
    for key in ("initial_run_result", "media_run_result", "autocred_run_result"):
        run_result = candidate.get(key) or {}
        output_dir = run_result.get("output_dir")
        if output_dir:
            dirs.append(Path(output_dir))
    best_record = candidate.get("calibration_best_record") or {}
    best_run_result = best_record.get("run_result") or {}
    if best_run_result.get("output_dir"):
        dirs.append(Path(best_run_result["output_dir"]))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in dirs:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def ensure_managed_output_dir(path: Path, config: dict[str, Any]) -> Path:
    resolved = path.resolve()
    output_root = (PROJECT_DIR / "outputs").resolve()
    promoted_root = (
        PROJECT_DIR
        / config.get("calibration", {}).get("promoted_workspace_dir", "model_store")
    ).resolve()
    if resolved == output_root:
        raise RuntimeError(f"Refusing to delete outputs root: {resolved}")
    if output_root in resolved.parents:
        return resolved

    # Successful calibrations are promoted into model_store and their
    # configured output directory is compacted in place after AutoCred.
    output_dir_name = str(config.get("output_dir_name", ".images"))
    if promoted_root in resolved.parents and resolved.name == output_dir_name:
        return resolved

    raise RuntimeError(
        f"Refusing to clean output outside managed roots "
        f"({output_root}, {promoted_root}/*/{output_dir_name}): {resolved}"
    )


def delete_candidate_outputs(candidate: dict[str, Any], config: dict[str, Any]) -> None:
    cleanup_config = config.get("outputs_cleanup", {})
    if not cleanup_config.get("delete_non_active_after_archive", False):
        return
    for output_dir in output_dirs_for_candidate(candidate):
        resolved = ensure_managed_output_dir(output_dir, config)
        if resolved.exists():
            shutil.rmtree(resolved)
    candidate["outputs_cleaned_after_archive"] = True


def compact_candidate_outputs(candidate: dict[str, Any], config: dict[str, Any]) -> None:
    cleanup_config = config.get("outputs_cleanup", {})
    if not cleanup_config.get("compact_active_after_autocred", False):
        return
    keep_metrics = bool(cleanup_config.get("keep_metrics", True))
    keep_logs = bool(cleanup_config.get("keep_logs", False))
    keep_mp4 = bool(cleanup_config.get("keep_mp4", False))
    keep_large_png = bool(cleanup_config.get("keep_large_png", False))
    keep_frames = bool(cleanup_config.get("keep_extracted_frames", True))
    for output_dir in output_dirs_for_candidate(candidate):
        resolved = ensure_managed_output_dir(output_dir, config)
        if not resolved.exists():
            continue
        if not keep_logs:
            shutil.rmtree(resolved / "logs", ignore_errors=True)
        if not keep_frames:
            shutil.rmtree(resolved / "frames", ignore_errors=True)
            if isinstance(candidate.get("media_visual"), dict):
                candidate["media_visual"]["frames"] = []
        images_dir = resolved / "images"
        if images_dir.is_dir():
            for path in images_dir.iterdir():
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                if suffix == ".mp4" and not keep_mp4:
                    path.unlink(missing_ok=True)
                elif suffix in {".png", ".jpg", ".jpeg"} and not keep_large_png:
                    path.unlink(missing_ok=True)
                elif not keep_metrics and path.name.endswith("_metrics.json"):
                    path.unlink(missing_ok=True)
    candidate["outputs_compacted_after_autocred"] = True


def summarize_metrics(metrics: dict[str, Any]) -> str:
    keys = [
        "goal_reach_rate",
        "average_completion_time",
        "completion_time",
        "average_speed",
        "total_distance_traveled",
        "average_distance_traveled",
        "stuck_pedestrian_count",
        "unique_overlap_count",
        "overlap_episode_count",
        "average_overlap_duration_seconds",
        "average_overlap_duration_per_pair_seconds",
        "total_overlap_duration_seconds",
        "overlap_count",
        "execution_time_seconds",
        "per_example_execution_time_seconds",
        "lines_of_code",
        "standard_cyclomatic_complexity",
        "halstead_effort",
    ]
    summary = {key: format_summary_value(metrics.get(key)) for key in keys if key in metrics}
    return json.dumps(summary, sort_keys=True)


def summarize_scenario_metrics(metrics: dict[str, Any]) -> str:
    scenario_metrics = metrics.get("scenario_metrics") or []
    if not scenario_metrics:
        return "No per-scenario metrics found."
    summaries: list[dict[str, Any]] = []
    keys = [
        "scenario_name",
        "goal_reach_rate",
        "average_completion_time",
        "completion_time",
        "average_speed",
        "total_distance_traveled",
        "average_distance_traveled",
        "stuck_pedestrian_count",
        "unique_overlap_count",
        "overlap_episode_count",
        "average_overlap_duration_seconds",
        "average_overlap_duration_per_pair_seconds",
        "total_overlap_duration_seconds",
        "overlap_count",
        "execution_time_seconds",
        "total_pedestrians",
        "reached_pedestrian_count",
    ]
    for item in scenario_metrics:
        summaries.append({key: format_summary_value(item.get(key)) for key in keys if key in item})
    return json.dumps(summaries, sort_keys=True)


def summarize_replay_result(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result.get("metrics", {})
    keys = [
        "goal_reach_rate",
        "reached_pedestrian_count",
        "average_completion_time",
        "average_speed",
        "total_distance_traveled",
        "average_distance_traveled",
        "stuck_pedestrian_count",
        "unique_overlap_count",
        "overlap_episode_count",
        "average_overlap_duration_seconds",
        "average_overlap_duration_per_pair_seconds",
        "total_overlap_duration_seconds",
        "overlap_count",
        "execution_time_seconds",
    ]
    return {
        "candidate_number": result.get("candidate_number"),
        "candidate_id": result.get("candidate_id"),
        "examples": result.get("examples"),
        "parameters": result.get("parameters"),
        "metrics": {key: format_summary_value(metrics.get(key)) for key in keys if key in metrics},
        "workspace": result.get("workspace"),
        "output_dir": result.get("output_dir"),
        "save_media": result.get("save_media"),
    }


def format_summary_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, float):
        return f"{value:.2f}"
    return value


def compute_complexity_snapshot(
    workspace: Path,
    config: dict[str, Any],
    label: str,
) -> dict[str, Any]:
    software_config = config.get("software_complexity", {})
    start_path = str(
        software_config.get(
            "start_path",
            config.get("model_file_name", "jsfm/sfm.py"),
        )
    )
    start_function = str(
        software_config.get("start_function", "single_update_active")
    ).strip()
    model_path = workspace / start_path
    existing_paths = [model_path] if model_path.exists() and model_path.suffix.lower() == ".py" else []
    snapshot: dict[str, Any] = {
        "label": label,
        "workspace": str(workspace),
        "files": [path.relative_to(workspace).as_posix() for path in existing_paths],
        "metrics": {},
        "available": False,
        "errors": [],
    }
    if not existing_paths:
        snapshot["errors"].append("No configured complexity files were found.")
        return snapshot

    try:
        from software_complexity_analyzer import analyze_software_complexity

        software_report = analyze_software_complexity(
            workspace,
            start_path,
            start_function,
        )
        resolved_scope = software_report["resolved_project_scope"]
        snapshot["software_complexity_summary"] = {
            "start_path": software_report.get("start_path"),
            "start_function": software_report.get("start_function"),
            "function_count": resolved_scope.get("function_count"),
            "reachable_helper_function_count": resolved_scope.get(
                "reachable_helper_function_count"
            ),
            "standard_cyclomatic_complexity": resolved_scope.get(
                "standard_cyclomatic_complexity"
            ),
            "halstead_effort": (resolved_scope.get("halstead") or {}).get("effort"),
        }
        snapshot["metrics"].update(
            {
                "standard_cyclomatic_complexity": snapshot[
                    "software_complexity_summary"
                ]["standard_cyclomatic_complexity"],
                "halstead_effort": snapshot["software_complexity_summary"][
                    "halstead_effort"
                ],
            }
        )
    except BaseException as exc:  # noqa: BLE001 - complexity is an acceptance guard, not a crash point.
        snapshot["errors"].append(f"Software complexity unavailable: {type(exc).__name__}: {exc}")

    snapshot["metrics"] = {
        key: value
        for key, value in snapshot["metrics"].items()
        if value is not None
    }
    snapshot["available"] = bool(snapshot["metrics"])
    return snapshot


def summarize_complexity_snapshot(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return "No complexity snapshot."
    if not snapshot.get("available"):
        return f"Unavailable: {snapshot.get('errors', [])}"
    return json.dumps(snapshot.get("metrics", {}), sort_keys=True)


def merge_software_complexity(metrics: dict[str, Any], complexity: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(metrics)
    if complexity:
        for key in ("standard_cyclomatic_complexity", "halstead_effort"):
            value = complexity.get("metrics", {}).get(key)
            if value is not None:
                merged[key] = value
    return merged


def autocred_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("autocred", {}).get("enabled", False))


def autocred_mode(config: dict[str, Any]) -> str:
    return str(config.get("autocred", {}).get("mode", "timeseries")).strip().lower()


def needs_openai_api_key(config: dict[str, Any]) -> bool:
    reasoning_backend = config.get("reasoning_backend", config.get("llm_backend", "codex_cli"))
    return reasoning_backend == "openai_api" or autocred_enabled(config)


def autocred_state_path(config: dict[str, Any]) -> Path:
    cfg = config.get("autocred", {})
    state_dir = Path(str(cfg.get("state_dir", "history/autocrd")))
    if not state_dir.is_absolute():
        state_dir = PROJECT_DIR / state_dir
    state_name = "autocrd_code_video_state.json" if autocred_mode(config) == "code_video" else "autocrd_state.json"
    return state_dir / state_name


def load_autocred_description(config: dict[str, Any]) -> str:
    cfg = config.get("autocred", {})
    inline = cfg.get("simulation_description")
    if inline:
        return str(inline)
    description_file = cfg.get("simulation_description_file")
    if description_file:
        path = Path(str(description_file))
        if not path.is_absolute():
            path = PROJECT_DIR / path
        return path.read_text(encoding="utf-8")
    return (
        "Pedestrian simulation based on a Social Force Model. Agents should move "
        "toward their goals with smooth, physically plausible velocities while "
        "maintaining personal space, avoiding overlaps, respecting obstacles and "
        "walls, and preserving interaction mechanisms consistent with social, "
        "driving, obstacle, and contact forces."
    )


def candidate_timeseries_name(config: dict[str, Any]) -> str:
    cfg = config.get("autocred", {})
    return str(cfg.get("timeseries_file_name") or config.get("trajectory_file_name", "trajectories.csv"))


def find_candidate_timeseries(candidate: dict[str, Any], config: dict[str, Any]) -> Path | None:
    file_name = candidate_timeseries_name(config)
    candidate_paths: list[Path] = []
    for run_key in ("autocred_run_result", "media_run_result", "initial_run_result"):
        run_result = candidate.get(run_key) or {}
        output_dir = run_result.get("output_dir")
        if output_dir:
            candidate_paths.extend(
                [
                    Path(output_dir) / file_name,
                    Path(output_dir) / "images" / file_name,
                ]
            )

    best_record = candidate.get("calibration_best_record") or {}
    best_run_result = best_record.get("run_result") or {}
    best_output_dir = best_run_result.get("output_dir")
    if best_output_dir:
        candidate_paths.append(Path(best_output_dir) / file_name)

    workspace = candidate.get("workspace")
    if workspace:
        candidate_paths.extend(
            [
                Path(workspace) / file_name,
                Path(workspace) / config.get("output_dir_name", ".images") / file_name,
            ]
        )

    for path in candidate_paths:
        if path.is_file():
            return path
    return None


def autocred_frame_paths(candidate: dict[str, Any], config: dict[str, Any]) -> list[Path]:
    if not config.get("send_video_frames_to_llm", True):
        return []
    frames = candidate.get("media_visual", {}).get("frames", [])
    return [Path(frame) for frame in frames if Path(frame).is_file()]


def ensure_candidate_timeseries(
    candidate: dict[str, Any],
    runner: SimulationRunner,
    config: dict[str, Any],
    examples: list[int] | None,
) -> Path | None:
    existing = find_candidate_timeseries(candidate, config)
    if existing:
        return existing
    if candidate.get("status") != "active":
        return None

    workspace = Path(candidate["workspace"])
    checkpoint(f"{candidate['candidate_id']}: running AutoCred time-series simulation", detail=True)
    run_result = runner.run(
        workspace,
        f"{candidate['candidate_id']}_autocred_timeseries",
        examples,
        PROJECT_DIR / "outputs",
        env_overrides={
            "SFM_SAVE_MEDIA": "0",
            "SFM_SAVE_PNG": "0",
            "SFM_SAVE_MP4": "0",
            "SFM_SAVE_TIMESERIES": "1",
        },
    )
    candidate["autocred_run_result"] = run_result.to_dict()
    if run_result.crashed or run_result.timed_out:
        return None
    return find_candidate_timeseries(candidate, config)


def bootstrap_autocred_if_needed(
    base_candidate: dict[str, Any],
    config: dict[str, Any],
    runner: SimulationRunner,
    examples: list[int] | None,
    evaluator: Evaluator | None = None,
    video_analyzer: VideoAnalyzer | None = None,
    baseline_lines: int = 0,
) -> dict[str, Any] | None:
    if not autocred_enabled(config):
        return None
    state_file = autocred_state_path(config)
    if state_file.is_file():
        checkpoint(f"Using existing AutoCred fixed rubric: {state_file}")
        return json.loads(state_file.read_text(encoding="utf-8"))

    cfg = config.get("autocred", {})
    state_dir = state_file.parent
    if autocred_mode(config) == "code_video":
        if not (base_candidate.get("media_visual") and base_candidate.get("media_run_result")):
            if evaluator is None or video_analyzer is None:
                raise RuntimeError("AutoCred code_video bootstrap requires base media frames.")
            base_candidate = generate_media_for_candidate(
                base_candidate,
                runner,
                evaluator,
                video_analyzer,
                config,
                examples,
                baseline_lines,
            )
        frames = autocred_frame_paths(base_candidate, config)
        if not frames:
            raise RuntimeError(
                "AutoCred code_video bootstrap could not find base visual frames "
                "(no extracted MP4 frames or fallback PNG images)."
            )
        source_context = read_code_excerpt(Path(base_candidate["workspace"]), config, None)

        from autocrd import bootstrap_autocred_code_video_rubric

        checkpoint("Bootstrapping fixed AutoCred code/video rubric from optimized base")
        state = bootstrap_autocred_code_video_rubric(
            simulation_description=load_autocred_description(config),
            source_context=source_context,
            visual_summary=base_candidate.get("media_visual", {}),
            image_paths=frames,
            state_dir=state_dir,
            model=str(cfg.get("model") or config.get("openai_model", "gpt-5-mini")),
            indicator_count=int(cfg.get("num_indicators", 4)),
        )
        base_candidate["autocred_bootstrap"] = {
            "mode": "code_video",
            "state_path": str(state_file),
            "indicator_count": len(state.get("indicators", [])),
            "frame_count": len(frames),
        }
        checkpoint(f"AutoCred fixed code/video rubric saved: {state_file}")
        return state

    timeseries_path = ensure_candidate_timeseries(base_candidate, runner, config, examples)
    if not timeseries_path:
        raise RuntimeError(
            "AutoCred is enabled, but no base time-series file was found or generated. "
            f"Expected file name: {candidate_timeseries_name(config)}"
        )

    from autocrd import bootstrap_autocred_rubric

    checkpoint(f"Bootstrapping fixed AutoCred rubric from optimized base time series: {timeseries_path}")
    state = bootstrap_autocred_rubric(
        csv_path=timeseries_path,
        simulation_description=load_autocred_description(config),
        state_dir=state_dir,
        model=str(cfg.get("model") or config.get("openai_model", "gpt-5-mini")),
        run_column=cfg.get("run_column"),
        parameter_columns=list(cfg.get("parameter_columns", [])),
        indicator_count=int(cfg.get("num_indicators", 4)),
        graph_count=int(cfg.get("num_graphs", 3)),
        max_sample_rows=int(cfg.get("max_sample_rows", 250)),
    )
    base_candidate["autocred_bootstrap"] = {
        "state_path": str(state_file),
        "source_timeseries": str(timeseries_path),
        "indicator_count": len(state.get("indicators", [])),
        "graph_count": state.get("graph_count"),
    }
    checkpoint(f"AutoCred fixed rubric saved: {state_file}")
    return state


def run_autocred_plausibility_gate(
    candidate: dict[str, Any],
    config: dict[str, Any],
    runner: SimulationRunner,
    evaluator: Evaluator,
    video_analyzer: VideoAnalyzer,
    examples: list[int] | None,
    baseline_lines: int,
) -> dict[str, Any]:
    if not autocred_enabled(config) or candidate.get("status") != "active":
        return candidate
    cfg = config.get("autocred", {})
    threshold = float(cfg.get("plausibility_threshold", 0.65))
    if autocred_mode(config) == "code_video":
        if not (candidate.get("media_visual") and candidate.get("media_run_result")):
            candidate = generate_media_for_candidate(
                candidate,
                runner,
                evaluator,
                video_analyzer,
                config,
                examples,
                baseline_lines,
            )
        frames = autocred_frame_paths(candidate, config)
        if not frames:
            candidate["status"] = "failed"
            candidate["failure_reason"] = (
                "AutoCred code/video plausibility check could not run because no visual "
                "frames were found (no extracted MP4 frames or fallback PNG images)."
            )
            candidate["autocred_plausibility"] = {
                "mode": "code_video",
                "status": "failed",
                "reason": candidate["failure_reason"],
                "threshold": threshold,
            }
            checkpoint(f"{candidate['candidate_id']}: AutoCred failed because video frames are missing")
            return candidate

        state_file = autocred_state_path(config)
        if not state_file.is_file():
            raise RuntimeError(f"AutoCred fixed rubric state is missing: {state_file}")

        from autocrd import load_autocred_state, score_code_video_plausibility_with_fixed_rubric

        state = load_autocred_state(state_file)
        source_context = read_code_excerpt(Path(candidate["workspace"]), config, None)
        out_dir = state_file.parent / "candidates" / str(candidate["candidate_id"])
        checkpoint(f"{candidate['candidate_id']}: running AutoCred code/video plausibility check")
        result = score_code_video_plausibility_with_fixed_rubric(
            source_context=source_context,
            visual_summary=candidate.get("media_visual", {}),
            metrics=candidate.get("metrics", {}),
            image_paths=frames,
            state=state,
            out_dir=out_dir,
            model=str(cfg.get("model") or state.get("model") or config.get("openai_model", "gpt-5-mini")),
            plausibility_threshold=threshold,
            run_face_validation=bool(cfg.get("run_face_validation", True)),
        )
        candidate["autocred_plausibility"] = {
            "mode": "code_video",
            "status": result.get("status"),
            "score": result.get("plausibility_score"),
            "threshold": result.get("plausibility_threshold"),
            "feedback_summary": result.get("feedback_summary"),
            "report_path": str(out_dir / "plausibility_report.json"),
            "source_context_path": str(out_dir / "source_context.txt"),
            "frame_count": len(frames),
        }
        if result.get("status") != "passed":
            candidate["status"] = "failed"
            candidate["failure_reason"] = (
                "AutoCred code/video plausibility score "
                f"{float(result.get('plausibility_score', 0.0)):.3f} below threshold {threshold:.3f}."
            )
            checkpoint(f"{candidate['candidate_id']}: AutoCred code/video plausibility failed")
        else:
            checkpoint(
                f"{candidate['candidate_id']}: AutoCred code/video plausibility passed "
                f"(score={float(result.get('plausibility_score', 0.0)):.3f})"
            )
        return candidate

    timeseries_path = ensure_candidate_timeseries(candidate, runner, config, examples)
    if not timeseries_path:
        candidate["status"] = "failed"
        candidate["failure_reason"] = (
            "AutoCred plausibility check could not run because no time-series file "
            f"was found or generated ({candidate_timeseries_name(config)})."
        )
        candidate["autocred_plausibility"] = {
            "status": "failed",
            "reason": candidate["failure_reason"],
            "threshold": threshold,
        }
        checkpoint(f"{candidate['candidate_id']}: AutoCred failed because time-series data is missing")
        return candidate

    state_file = autocred_state_path(config)
    if not state_file.is_file():
        raise RuntimeError(f"AutoCred fixed rubric state is missing: {state_file}")

    from autocrd import load_autocred_state, score_plausibility_with_fixed_rubric

    state = load_autocred_state(state_file)
    out_dir = state_file.parent / "candidates" / str(candidate["candidate_id"])
    checkpoint(f"{candidate['candidate_id']}: running AutoCred plausibility check")
    result = score_plausibility_with_fixed_rubric(
        csv_path=timeseries_path,
        state=state,
        out_dir=out_dir,
        model=str(cfg.get("model") or state.get("model") or config.get("openai_model", "gpt-5-mini")),
        run_column=cfg.get("run_column"),
        parameter_columns=list(cfg.get("parameter_columns", [])),
        plausibility_threshold=threshold,
        run_face_validation=bool(cfg.get("run_face_validation", True)),
    )
    candidate["autocred_plausibility"] = {
        "status": result.get("status"),
        "score": result.get("plausibility_score"),
        "threshold": result.get("plausibility_threshold"),
        "feedback_summary": result.get("feedback_summary"),
        "report_path": str(out_dir / "plausibility_report.json"),
        "source_timeseries": str(timeseries_path),
    }
    if result.get("status") != "passed":
        candidate["status"] = "failed"
        candidate["failure_reason"] = (
            "AutoCred plausibility score "
            f"{float(result.get('plausibility_score', 0.0)):.3f} below threshold {threshold:.3f}."
        )
        checkpoint(f"{candidate['candidate_id']}: AutoCred plausibility failed")
    else:
        checkpoint(
            f"{candidate['candidate_id']}: AutoCred plausibility passed "
            f"(score={float(result.get('plausibility_score', 0.0)):.3f})"
        )
    return candidate


def create_child_from_parent(
    parent: dict[str, Any],
    iteration: int,
    child_index: int,
    attempt: int,
    llm: LLMClient,
    history: HistoryManager,
    patch_manager: PatchManager,
    config: dict[str, Any],
    run_examples: list[int] | None,
    plausibility_failure_feedback: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parent_workspace = Path(parent["workspace"])
    parent_visual = parent.get("media_visual", {})
    parent_metrics = parent.get("metrics", {})
    parent_frames = [
        Path(frame)
        for frame in parent_visual.get("frames", [])
        if config.get("send_video_frames_to_llm", True)
    ]
    code_excerpt = read_code_excerpt(
        parent_workspace,
        config,
        None,
    )
    child_id = child_candidate_id(iteration, parent, child_index, attempt=attempt)
    checkpoint(
        f"Asking OpenAI for replacement child {child_id} from parent {parent['candidate_id']}"
        if attempt > 1
        else f"Asking OpenAI for child {child_id} from parent {parent['candidate_id']}"
    )
    parent_state = pool_candidate_summary(parent)
    if plausibility_failure_feedback:
        parent_state["recent_plausibility_failure_feedback"] = plausibility_failure_feedback
    diagnosis = llm.diagnose(
        iteration=iteration,
        workspace=parent_workspace,
        examples=run_examples,
        metrics=parent_metrics,
        # Keep the complete history on disk, but send only the latest compact
        # entry so accumulated pool bookkeeping cannot exceed the model context.
        history=diagnosis_history_context(history, config),
        visual_summary=parent_visual,
        code_excerpt=code_excerpt,
        max_children=1,
        parent_state=parent_state,
        image_paths=parent_frames,
    )
    selected = select_child_candidates(diagnosis, 1)
    if not selected:
        return None, diagnosis

    llm_child = selected[0]
    parent["has_produced_child"] = True
    parent["produced_child_count"] = int(parent.get("produced_child_count", 0) or 0) + 1
    parent["last_produced_iteration"] = iteration
    checkpoint(f"Creating child workspace {child_id}: {llm_child.get('name', 'unnamed child')}")
    child_workspace = patch_manager.create_candidate_workspace(
        parent_workspace,
        iteration,
        run_examples,
    )
    if not config.get("automatic_patch_approval", True):
        terminal_print(json.dumps(llm_child, indent=2))
        answer = timestamped_input(f"Apply child {child_id} with Codex CLI? [y/N] ").strip().lower()
        if answer != "y":
            checkpoint(f"Skipping child {child_id} before patch application.")
            return None, diagnosis

    edit_scope_baseline = patch_manager.capture_edit_scope(child_workspace)
    checkpoint(f"{child_id}: starting Codex CLI edit")
    llm.apply_with_codex(
        iteration=iteration,
        candidate_workspace=child_workspace,
        candidate=llm_child,
        metrics=parent_metrics,
        failed_history=diagnosis_history_context(history, config),
        code_excerpt=code_excerpt,
    )
    checkpoint(f"{child_id}: Codex CLI edit completed")

    checkpoint(f"{child_id}: validating patched workspace", detail=True)
    edit_scope_validation = patch_manager.validate_edit_scope(
        child_workspace,
        edit_scope_baseline,
    )
    function_scope_validation = patch_manager.validate_sfm_function_scope(
        parent_workspace,
        child_workspace,
    )
    validation = patch_manager.validate(child_workspace)
    validation["edit_scope"] = edit_scope_validation
    validation["function_scope"] = function_scope_validation
    if not edit_scope_validation["ok"]:
        validation["ok"] = False
        validation["edit_scope_error"] = edit_scope_validation["error"]
    if not function_scope_validation["ok"]:
        validation["ok"] = False
        validation["function_scope_error"] = function_scope_validation["error"]
    try:
        validation["calibration_manifest_inheritance"] = validate_manifest_inheritance(
            parent_workspace,
            child_workspace,
            config,
        )
    except (OSError, ValueError) as exc:
        validation["ok"] = False
        validation["calibration_manifest_inheritance_error"] = str(exc)
    checkpoint(f"{child_id}: validation result ok={validation.get('ok')}")
    patch_text = patch_manager.diff(parent_workspace, child_workspace)
    child = {
        "candidate_id": child_id,
        "parent_id": parent["candidate_id"],
        "iteration_created": iteration,
        "has_produced_child": False,
        "produced_child_count": 0,
        "status": "active" if validation.get("ok") else "failed",
        "workspace": str(child_workspace),
        "calibration_manifest": str(child_workspace / CALIBRATION_MANIFEST_NAME),
        "parent_workspace": str(parent_workspace),
        "llm_candidate": llm_child,
        "llm_diagnosis": diagnosis.get("diagnosis", ""),
        "validation": validation,
        "patch": patch_text,
        "child_slot_index": child_index,
        "replacement_attempt": attempt,
    }
    if not validation.get("ok"):
        child["failure_reason"] = (
            validation.get("edit_scope_error")
            or validation.get("function_scope_error")
            or validation.get("calibration_manifest_error")
            or validation.get("calibration_manifest_inheritance_error")
            or "Validation failed before simulation."
        )
        checkpoint(f"{child_id}: validation failed; child will not be simulated")
    return child, diagnosis


def diagnose_parent_for_children(
    *,
    parent: dict[str, Any],
    child_count: int,
    iteration: int,
    llm: LLMClient,
    history_context: list[dict[str, Any]],
    config: dict[str, Any],
    run_examples: list[int] | None,
) -> dict[str, Any]:
    parent_workspace = Path(parent["workspace"])
    parent_visual = parent.get("media_visual", {})
    parent_metrics = parent.get("metrics", {})
    parent_frames = [
        Path(frame)
        for frame in parent_visual.get("frames", [])
        if config.get("send_video_frames_to_llm", True)
    ]
    checkpoint(f"Preparing LLM context for parent {parent['candidate_id']}", detail=True)
    code_excerpt = read_code_excerpt(parent_workspace, config, None)
    checkpoint(
        f"Parent {parent['candidate_id']} context: "
        f"{len(code_excerpt)} code chars, {len(parent_frames)} frame(s), "
        f"metrics={summarize_metrics(parent_metrics)}",
        detail=True,
    )
    checkpoint(
        f"Asking OpenAI for exactly {child_count} child pseudocode option(s) "
        f"from parent {parent['candidate_id']}"
    )
    diagnosis = llm.diagnose(
        iteration=iteration,
        workspace=parent_workspace,
        examples=run_examples,
        metrics=parent_metrics,
        history=history_context,
        visual_summary=parent_visual,
        code_excerpt=code_excerpt,
        max_children=child_count,
        parent_state=pool_candidate_summary(parent),
        image_paths=parent_frames,
    )
    checkpoint(
        f"OpenAI returned diagnosis for parent {parent['candidate_id']} with "
        f"{len(diagnosis.get('candidates', []))} candidate option(s)",
        detail=True,
    )
    return {
        "parent": parent,
        "child_count": child_count,
        "parent_workspace": parent_workspace,
        "parent_metrics": parent_metrics,
        "code_excerpt": code_excerpt,
        "diagnosis": diagnosis,
    }


def run_diagnosis_jobs(
    prepared_jobs: list[dict[str, Any]],
    configured_workers: Any,
) -> list[dict[str, Any]]:
    worker_count = resolve_parallel_workers(configured_workers, len(prepared_jobs))
    if not prepared_jobs:
        return []

    completed_results: list[dict[str, Any] | None] = [None] * len(prepared_jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(diagnose_parent_for_children, **job): index
            for index, job in enumerate(prepared_jobs)
        }
        for future in concurrent.futures.as_completed(futures):
            completed_results[futures[future]] = future.result()
    return [result for result in completed_results if result is not None]


def apply_codex_edit_and_build_child(
    *,
    child_id: str,
    parent: dict[str, Any],
    iteration: int,
    child_index: int,
    llm_child: dict[str, Any],
    diagnosis: dict[str, Any],
    child_workspace: Path,
    edit_scope_baseline: dict[str, Any],
    parent_metrics: dict[str, Any],
    code_excerpt: str,
    failed_history: list[dict[str, Any]],
    llm: LLMClient,
    patch_manager: PatchManager,
    config: dict[str, Any],
) -> dict[str, Any]:
    parent_workspace = Path(parent["workspace"])
    checkpoint(f"{child_id}: starting Codex CLI edit")
    llm.apply_with_codex(
        iteration=iteration,
        candidate_workspace=child_workspace,
        candidate=llm_child,
        metrics=parent_metrics,
        failed_history=failed_history,
        code_excerpt=code_excerpt,
    )
    checkpoint(f"{child_id}: Codex CLI edit completed")

    checkpoint(f"{child_id}: validating patched workspace", detail=True)
    edit_scope_validation = patch_manager.validate_edit_scope(
        child_workspace,
        edit_scope_baseline,
    )
    function_scope_validation = patch_manager.validate_sfm_function_scope(
        parent_workspace,
        child_workspace,
    )
    validation = patch_manager.validate(child_workspace)
    validation["edit_scope"] = edit_scope_validation
    validation["function_scope"] = function_scope_validation
    if not edit_scope_validation["ok"]:
        validation["ok"] = False
        validation["edit_scope_error"] = edit_scope_validation["error"]
    if not function_scope_validation["ok"]:
        validation["ok"] = False
        validation["function_scope_error"] = function_scope_validation["error"]
    try:
        validation["calibration_manifest_inheritance"] = validate_manifest_inheritance(
            parent_workspace,
            child_workspace,
            config,
        )
    except (OSError, ValueError) as exc:
        validation["ok"] = False
        validation["calibration_manifest_inheritance_error"] = str(exc)
    checkpoint(f"{child_id}: validation result ok={validation.get('ok')}")
    checkpoint(f"{child_id}: collecting diff against parent workspace", detail=True)
    patch_text = patch_manager.diff(parent_workspace, child_workspace)
    checkpoint(f"{child_id}: diff length {len(patch_text)} character(s)", detail=True)
    child = {
        "candidate_id": child_id,
        "parent_id": parent["candidate_id"],
        "iteration_created": iteration,
        "has_produced_child": False,
        "produced_child_count": 0,
        "status": "active" if validation.get("ok") else "failed",
        "workspace": str(child_workspace),
        "calibration_manifest": str(child_workspace / CALIBRATION_MANIFEST_NAME),
        "parent_workspace": str(parent_workspace),
        "llm_candidate": llm_child,
        "llm_diagnosis": diagnosis.get("diagnosis", ""),
        "validation": validation,
        "patch": patch_text,
        "child_slot_index": child_index,
        "replacement_attempt": 1,
    }
    if not validation.get("ok"):
        child["failure_reason"] = (
            validation.get("edit_scope_error")
            or validation.get("function_scope_error")
            or validation.get("calibration_manifest_error")
            or validation.get("calibration_manifest_inheritance_error")
            or "Validation failed before simulation."
        )
        checkpoint(f"{child_id}: validation failed; child will not be simulated")
    return child


def run_codex_edit_jobs(
    prepared_jobs: list[dict[str, Any]],
    configured_workers: Any,
) -> list[dict[str, Any]]:
    worker_count = resolve_parallel_workers(configured_workers, len(prepared_jobs))
    if not prepared_jobs:
        return []

    completed_children: list[dict[str, Any] | None] = [None] * len(prepared_jobs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(apply_codex_edit_and_build_child, **job): index
            for index, job in enumerate(prepared_jobs)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            child = future.result()
            completed_children[index] = child
            checkpoint(
                f"{child['candidate_id']}: added to this iteration's child list "
                f"with status={child['status']}",
                detail=True,
            )
    return [child for child in completed_children if child is not None]


def run_single_child_lifecycle(
    child: dict[str, Any],
    config: dict[str, Any],
    runner: SimulationRunner,
    evaluator: Evaluator,
    video_analyzer: VideoAnalyzer,
    examples: list[int] | None,
    baseline_lines: int,
    concurrent_model_count: int,
) -> dict[str, Any]:
    child = run_child_metrics_only(child, runner, evaluator, config, examples, baseline_lines)
    if child.get("status") != "active":
        return child
    child = calibrate_child_candidate(
        child,
        config,
        runner,
        evaluator,
        examples,
        concurrent_model_count,
    )
    if child.get("status") != "active":
        return child
    child = run_autocred_plausibility_gate(
        child,
        config,
        runner,
        evaluator,
        video_analyzer,
        examples,
        baseline_lines,
    )
    compact_candidate_outputs(child, config)
    return child


def retry_autocred_failed_children(
    children: list[dict[str, Any]],
    candidate_pool: list[dict[str, Any]],
    iteration: int,
    llm: LLMClient,
    history: HistoryManager,
    patch_manager: PatchManager,
    config: dict[str, Any],
    runner: SimulationRunner,
    evaluator: Evaluator,
    video_analyzer: VideoAnalyzer,
    run_examples: list[int] | None,
    baseline_lines: int,
    diagnoses: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not autocred_enabled(config):
        return children
    max_attempts = max(1, int(config.get("autocred", {}).get("max_replacement_attempts", 3)))
    by_id = {candidate.get("candidate_id"): candidate for candidate in candidate_pool + children}
    final_children = list(children)
    failed_slots = [
        child
        for child in children
        if child.get("status") == "failed"
        and (
            "AutoCred plausibility" in str(child.get("failure_reason", ""))
            or "AutoCred code/video plausibility" in str(child.get("failure_reason", ""))
            or (child.get("autocred_plausibility") or {}).get("status") == "failed"
        )
    ]

    for failed_child in failed_slots:
        parent = by_id.get(failed_child.get("parent_id"))
        if not parent:
            continue
        slot_index = int(failed_child.get("child_slot_index") or 1)
        accepted_replacement = None
        for attempt in range(2, max_attempts + 1):
            checkpoint(
                f"{failed_child['candidate_id']}: generating AutoCred replacement "
                f"attempt {attempt}/{max_attempts}"
            )
            plausibility_feedback = {
                "failed_candidate_id": failed_child.get("candidate_id"),
                "status": (failed_child.get("autocred_plausibility") or {}).get("status"),
                "score": (failed_child.get("autocred_plausibility") or {}).get("score"),
                "threshold": (failed_child.get("autocred_plausibility") or {}).get("threshold"),
                "feedback_summary": (failed_child.get("autocred_plausibility") or {}).get("feedback_summary"),
                "report_path": (failed_child.get("autocred_plausibility") or {}).get("report_path"),
                "instruction": (
                    "Generate a different single-mechanism child that directly addresses "
                    "this plausibility failure while preserving the parent model's useful behavior."
                ),
            }
            try:
                replacement, diagnosis = create_child_from_parent(
                    parent=parent,
                    iteration=iteration,
                    child_index=slot_index,
                    attempt=attempt,
                    llm=llm,
                    history=history,
                    patch_manager=patch_manager,
                    config=config,
                    run_examples=run_examples,
                    plausibility_failure_feedback=plausibility_feedback,
                )
            except Exception as exc:  # noqa: BLE001 - this is an infrastructure failure.
                checkpoint(f"Stopping safely: replacement child generation failed with {type(exc).__name__}: {exc}")
                raise
            if diagnosis:
                diagnoses.append(diagnosis)
            if not replacement:
                continue
            replacement["replaces_candidate_id"] = failed_child.get("candidate_id")
            final_children.append(replacement)
            replacement = run_single_child_lifecycle(
                replacement,
                config,
                runner,
                evaluator,
                video_analyzer,
                run_examples,
                baseline_lines,
                1,
            )
            final_children[-1] = replacement
            by_id[replacement.get("candidate_id")] = replacement
            if replacement.get("status") == "active":
                accepted_replacement = replacement
                break
        if accepted_replacement is None:
            failed_child["autocred_replacement_exhausted"] = True
            failed_child["autocred_replacement_attempts"] = max_attempts
            checkpoint(
                f"{failed_child['candidate_id']}: no plausible replacement after "
                f"{max_attempts} attempt(s)"
            )

    return final_children


def run_child_metrics_only(
    child: dict[str, Any],
    runner: SimulationRunner,
    evaluator: Evaluator,
    config: dict[str, Any],
    examples: list[int] | None,
    baseline_lines: int,
) -> dict[str, Any]:
    checkpoint(f"{child.get('candidate_id')}: metrics-only worker started", detail=True)
    workspace = Path(child["workspace"])
    validation = child.get("validation") or {"ok": False}
    if not validation.get("ok"):
        child["status"] = "failed"
        child["failure_reason"] = "Validation failed before simulation."
        checkpoint(f"{child.get('candidate_id')}: skipped metrics-only run because validation failed")
        return child

    run_id = f"{child['candidate_id']}_initial"
    checkpoint(f"{child['candidate_id']}: running metrics-only simulation with media disabled", detail=True)
    run_result = runner.run(
        workspace,
        run_id,
        examples,
        PROJECT_DIR / "outputs",
        env_overrides={"SFM_SAVE_MEDIA": "0"},
    )
    child["initial_run_result"] = run_result.to_dict()
    checkpoint(
        f"{child['candidate_id']}: metrics-only run finished "
        f"(crashed={run_result.crashed}, timed_out={run_result.timed_out})",
        detail=not (run_result.crashed or run_result.timed_out),
    )
    if run_result.crashed or run_result.timed_out:
        child["status"] = "failed"
        child["failure_reason"] = "Initial metrics-only simulation failed."
        return child

    checkpoint(f"{child['candidate_id']}: evaluating metrics-only output", detail=True)
    metrics = evaluator.evaluate(run_result.to_dict(), workspace, baseline_lines)
    checkpoint(f"{child['candidate_id']}: metrics summary {summarize_metrics(metrics)}")
    checkpoint(f"{child['candidate_id']}: computing static software complexity", detail=True)
    complexity = compute_complexity_snapshot(workspace, config, f"{child['candidate_id']}_initial")
    checkpoint(
        f"{child['candidate_id']}: complexity summary {summarize_complexity_snapshot(complexity)}",
        detail=True,
    )
    child["initial_metrics"] = merge_software_complexity(metrics, complexity)
    child["initial_complexity"] = complexity
    child["metrics"] = child["initial_metrics"]
    child["complexity"] = complexity
    return child


def calibrate_child_candidate(
    child: dict[str, Any],
    config: dict[str, Any],
    runner: SimulationRunner,
    evaluator: Evaluator,
    examples: list[int] | None,
    concurrent_model_count: int,
) -> dict[str, Any]:
    if child.get("status") != "active":
        checkpoint(f"{child.get('candidate_id')}: skipped calibration because status={child.get('status')}")
        return child

    checkpoint(f"{child['candidate_id']}: archive GA calibration worker started")
    pre_calibration_workspace = child.get("workspace")
    calibrator = build_calibrator(config, runner, evaluator)
    result = calibrator.run(
        Path(child["workspace"]),
        examples,
        concurrent_model_count=concurrent_model_count,
    )
    checkpoint(f"{child['candidate_id']}: archive GA calibration worker finished", detail=True)
    child["calibration_result"] = result
    best_path = result.get("best_calibration_path")
    if not best_path:
        child["status"] = "failed"
        child["failure_reason"] = "Calibration did not produce best_calibration.json."
        checkpoint(f"{child['candidate_id']}: calibration failed to produce best_calibration.json")
        return child

    checkpoint(f"{child['candidate_id']}: loading selected calibration from {best_path}", detail=True)
    best_record = json.loads(Path(best_path).read_text(encoding="utf-8"))
    best_workspace = Path(best_record["workspace"])
    if config.get("calibration", {}).get("cleanup_after_success", False):
        checkpoint(f"{child['candidate_id']}: promoting selected calibration workspace before cleanup", detail=True)
        original_run_dir = result.get("run_dir")
        promoted_workspace = promote_calibrated_workspace(child, best_workspace, config)
        best_workspace = promoted_workspace
        best_record["workspace"] = str(promoted_workspace)
        if isinstance(best_record.get("run_result"), dict):
            best_record["run_result"]["workspace"] = str(promoted_workspace)
            best_record["run_result"]["output_dir"] = str(promoted_workspace / config.get("output_dir_name", ".images"))
        if best_record.get("raw_metrics_path"):
            best_record["raw_metrics_path"] = str(
                promoted_workspace / config.get("output_dir_name", ".images") / "example10_metrics.json"
            )
        promoted_best_path = promoted_workspace / "best_calibration.json"
        promoted_best_path.write_text(json.dumps(best_record, indent=2), encoding="utf-8")
        result["original_run_dir"] = original_run_dir
        result["run_dir_cleaned"] = True
        result["promoted_workspace"] = str(promoted_workspace)
        result["best_calibration_path"] = str(promoted_best_path)
        result["calibration_manifest_path"] = str(promoted_workspace / CALIBRATION_MANIFEST_NAME)
        result["cleanup_note"] = "Heavy calibration run artifacts were deleted after the selected workspace was promoted."
        result.pop("candidate_pool_path", None)
        result.pop("removed_candidates_path", None)
        result.pop("plots_dir", None)
        result.pop("native_ga_best_calibration_path", None)
        result.pop("report_path", None)
        cleanup_calibration_run(result | {"run_dir": original_run_dir}, config)
        result["run_dir"] = None
        cleanup_disposable_workspace(pre_calibration_workspace, config)
    best_metrics = dict(best_record.get("metrics", {}))
    best_selected = dict(best_record.get("selected_metrics", {}))
    best_metrics.update(best_selected)
    child["workspace"] = str(best_workspace)
    child["calibrated_workspace"] = str(best_workspace)
    child["calibration_manifest"] = str(best_workspace / CALIBRATION_MANIFEST_NAME)
    child["calibration_best_record"] = best_record
    child["metrics"] = best_metrics
    child["complexity"] = {
        "label": f"{child['candidate_id']}_calibrated",
        "workspace": str(best_workspace),
        "metrics": {
            "standard_cyclomatic_complexity": best_metrics.get(
                "standard_cyclomatic_complexity"
            ),
            "halstead_effort": best_metrics.get("halstead_effort"),
        },
        "available": all(
            best_metrics.get(key) is not None
            for key in ("standard_cyclomatic_complexity", "halstead_effort")
        ),
        "errors": [],
    }
    checkpoint(f"{child['candidate_id']}: selected calibration metrics {summarize_metrics(best_metrics)}")
    return child


def generate_media_for_candidate(
    candidate: dict[str, Any],
    runner: SimulationRunner,
    evaluator: Evaluator,
    video_analyzer: VideoAnalyzer,
    config: dict[str, Any],
    examples: list[int] | None,
    baseline_lines: int,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    if candidate.get("status") != "active":
        checkpoint(
            f"{candidate.get('candidate_id')}: skipped media generation "
            f"(status={candidate.get('status')})"
        )
        return candidate
    if candidate.get("media_visual") and candidate.get("media_run_result"):
        checkpoint(f"{candidate.get('candidate_id')}: skipped media generation because media already exists")
        return candidate
    workspace = Path(candidate["workspace"])
    checkpoint(f"{candidate['candidate_id']}: media worker started", detail=True)
    run_result = runner.run(
        workspace,
        f"{candidate['candidate_id']}_media",
        examples,
        PROJECT_DIR / "outputs",
        env_overrides=env_overrides,
    )
    candidate["media_run_result"] = run_result.to_dict()
    checkpoint(
        f"{candidate['candidate_id']}: media run finished "
        f"(crashed={run_result.crashed}, timed_out={run_result.timed_out})",
        detail=not (run_result.crashed or run_result.timed_out),
    )
    if run_result.crashed or run_result.timed_out:
        candidate["media_visual"] = {
            "enabled": config.get("enable_video_analysis", True),
            "frames": [],
            "summary": "Media run failed after candidate survived comparison.",
        }
        return candidate
    checkpoint(f"{candidate['candidate_id']}: evaluating media run metrics", detail=True)
    media_metrics = evaluator.evaluate(run_result.to_dict(), workspace, baseline_lines)
    checkpoint(f"{candidate['candidate_id']}: media metrics summary {summarize_metrics(media_metrics)}", detail=True)
    checkpoint(f"{candidate['candidate_id']}: recomputing software complexity after media run", detail=True)
    media_complexity = compute_complexity_snapshot(workspace, config, f"{candidate['candidate_id']}_media")
    candidate["metrics"] = merge_software_complexity(media_metrics, media_complexity)
    candidate["complexity"] = media_complexity
    checkpoint(f"{candidate['candidate_id']}: extracting/analyzing video frames", detail=True)
    candidate["media_visual"] = video_analyzer.analyze_outputs(Path(run_result.output_dir), workspace)
    checkpoint(
        f"{candidate['candidate_id']}: media ready with "
        f"{len(candidate.get('media_visual', {}).get('frames', []))} frame(s)"
    )
    return candidate


def candidate_graph_labels(candidate_pool: list[dict[str, Any]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    next_index = 0
    for candidate in candidate_pool:
        candidate_id = str(candidate.get("candidate_id"))
        if candidate_id == "base":
            labels[candidate_id] = "base"
        else:
            labels[candidate_id] = spreadsheet_label(next_index)
            next_index += 1
    return labels


def spreadsheet_label(index: int) -> str:
    label = ""
    value = index
    while True:
        value, remainder = divmod(value, 26)
        label = chr(ord("A") + remainder) + label
        if value == 0:
            return label
        value -= 1


def write_iteration_graph(candidate_pool: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "iteration_candidate_graph.json"
    png_path = output_dir / "iteration_candidate_graph.png"
    labels = candidate_graph_labels(candidate_pool)
    graph_data = {
        "nodes": [
            {
                "candidate_id": candidate.get("candidate_id"),
                "display_label": labels.get(str(candidate.get("candidate_id")), str(candidate.get("candidate_id"))),
                "parent_id": candidate.get("parent_id"),
                "iteration_created": candidate.get("iteration_created"),
                "status": candidate.get("status"),
                "has_produced_child": candidate.get("has_produced_child"),
                "dominated_by": candidate.get("dominated_by"),
                "dominated_order": candidate.get("dominated_order"),
                "metrics": candidate.get("metrics", {}),
            }
            for candidate in candidate_pool
        ],
        "edges": [
            {
                "source": candidate.get("parent_id"),
                "target": candidate.get("candidate_id"),
            }
            for candidate in candidate_pool
            if candidate.get("parent_id")
        ],
    }
    json_path.write_text(json.dumps(graph_data, indent=2), encoding="utf-8")

    try:
        import matplotlib  # type: ignore
        import networkx as nx  # type: ignore

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        return {
            "json_path": str(json_path),
            "png_path": None,
            "error": f"Graph PNG skipped because dependency is unavailable: {type(exc).__name__}",
        }

    graph = nx.DiGraph()
    for node in graph_data["nodes"]:
        graph.add_node(
            node["candidate_id"],
            subset=int(node.get("iteration_created") or 0),
            **node,
        )
    for edge in graph_data["edges"]:
        if edge["source"] in graph and edge["target"] in graph:
            graph.add_edge(edge["source"], edge["target"])

    if not graph.nodes:
        return {"json_path": str(json_path), "png_path": None, "error": "No graph nodes."}

    position = nx.multipartite_layout(graph, subset_key="subset", align="vertical")
    status_colors = {
        "active": "#d9f7d9",
        "dominated": "#ffd2d2",
        "failed": "#e0e0e0",
    }
    node_colors = [
        status_colors.get(graph.nodes[node].get("status"), "#e8f1ff")
        for node in graph.nodes
    ]
    figure_width = max(10, 2 + len(graph.nodes) * 0.7)
    figure_height = max(6, 2 + len({data.get("subset") for _, data in graph.nodes(data=True)}) * 1.4)
    plt.figure(figsize=(figure_width, figure_height))
    nx.draw_networkx_edges(
        graph,
        position,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=14,
        width=1.2,
        edge_color="#6b7280",
    )
    nx.draw_networkx_nodes(
        graph,
        position,
        node_shape="o",
        node_size=1450,
        node_color=node_colors,
        edgecolors="#222222",
        linewidths=1.4,
    )

    inside_labels = {
        node: graph_node_label(graph.nodes[node])
        for node in graph.nodes
    }
    nx.draw_networkx_labels(
        graph,
        position,
        labels=inside_labels,
        font_size=12,
        font_weight="bold",
    )
    for node, (x_pos, y_pos) in position.items():
        status = graph.nodes[node].get("status")
        label = f"{graph.nodes[node].get('display_label')}\n{status}"
        plt.text(
            x_pos,
            y_pos - 0.13,
            label,
            ha="center",
            va="top",
            fontsize=7,
        )

    plt.title("Pool Refinement Candidate Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()
    return {"json_path": str(json_path), "png_path": str(png_path), "error": None}


def graph_node_label(node_data: dict[str, Any]) -> str:
    display_label = str(node_data.get("display_label") or node_data.get("candidate_id") or "")
    dominated_order = node_data.get("dominated_order")
    if dominated_order:
        return f"{display_label}\n{dominated_order}"
    return display_label


def to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_calibrator(
    config: dict[str, Any],
    runner: SimulationRunner,
    evaluator: Evaluator,
) -> ArchiveGACalibrator:
    return ArchiveGACalibrator(
        config,
        PROJECT_DIR,
        runner,
        evaluator,
        checkpoint=checkpoint,
    )


def main() -> int:
    global TERMINAL_VERBOSITY
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    config = load_config(config_path)
    TERMINAL_VERBOSITY = str(config.get("terminal_verbosity", "concise")).lower()
    if TERMINAL_VERBOSITY not in {"concise", "detailed"}:
        raise ValueError("terminal_verbosity must be 'concise' or 'detailed'.")
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    configured_max_iterations = int(config.get("max_iterations", 5))
    config.setdefault("calibration", {})
    if needs_openai_api_key(config):
        try:
            resolve_openai_api_key()
        except RuntimeError as exc:
            checkpoint(f"Stopping safely: {exc}")
            return 1

    examples = determine_examples(args, config)
    checkpoint("Loading configuration and preparing controllers")
    checkpoint(
        "Backends: reasoning="
        f"{config.get('reasoning_backend', config.get('llm_backend', 'codex_cli'))}, "
        f"code_edit={config.get('code_edit_backend', 'codex_cli')}",
        detail=True,
    )
    patch_manager = PatchManager(config, PROJECT_DIR)
    seed_export_source = (
        patch_manager.calibration_source(examples)
        if args.resume or args.calibration
        else patch_manager.source_project_dir
    )
    checkpoint(f"Exporting current seed files for inspection from: {seed_export_source}", detail=True)
    patch_manager.export_seed_files(seed_export_source, examples)
    runner = SimulationRunner(config, PROJECT_DIR)
    evaluator = Evaluator(config)
    video_analyzer = VideoAnalyzer(config)

    if args.run_calibration_candidate:
        replay = CalibrationCandidateRunner(
            config,
            PROJECT_DIR,
        runner,
            evaluator,
            checkpoint=checkpoint,
        )
        result = replay.run(
            candidate_number=args.run_calibration_candidate,
            examples=examples,
            calibration_run_dir=Path(args.calibration_run_dir) if args.calibration_run_dir else None,
            save_media=not args.candidate_no_media,
        )
        terminal_print(json.dumps(summarize_replay_result(result), indent=2))
        checkpoint(f"Calibration candidate replay done. Output: {result['output_dir']}")
        return 0

    history = HistoryManager(PROJECT_DIR / "history")
    llm = LLMClient(config, PROJECT_DIR, history)
    stored_pool_state = (
        history.load_pool_state()
        if args.resume and not args.calibration and not args.baseline_only
        else None
    )

    if args.calibration:
        checkpoint("Creating calibration seed workspace copy", detail=True)
        current_workspace = patch_manager.prepare_calibration_workspace(examples)
        seed_source = patch_manager.calibration_source(examples)
        checkpoint(f"Starting archive GA calibration for example10 on seed: {seed_source}")
        calibrator = build_calibrator(config, runner, evaluator)
        result = calibrator.run(current_workspace, examples)
        checkpoint(f"Archive GA calibration done. Report: {result['report_path']}")
        terminal_print(json.dumps({"calibration_result": result}, indent=2))
        return 0 if result.get("best_calibration_path") else 1

    candidate_pool: list[dict[str, Any]] = []
    domination_sequence = 0
    baseline_lines = 0
    run_examples = examples
    active_pool_state: dict[str, Any] | None = None
    resumed_from_pool = False

    if stored_pool_state:
        restored_pool, restored_domination_sequence, state_examples, state_baseline_lines = restore_pool_state(
            stored_pool_state
        )
        resume_workspace = first_active_workspace(restored_pool)
        if resume_workspace:
            candidate_pool = restored_pool
            domination_sequence = restored_domination_sequence
            run_examples = examples or state_examples or list(config.get("example_numbers", []))
            baseline_lines = state_baseline_lines or evaluator.count_lines_of_code(resume_workspace)
            active_pool_state = stored_pool_state
            resumed_from_pool = True
            checkpoint(
                "Resuming from saved pool state "
                f"{history.pool_state_path} after iteration "
                f"{stored_pool_state.get('last_completed_iteration', 0)}"
            )
            checkpoint(f"Seed inspection copy now points to active pool workspace: {resume_workspace}", detail=True)
            patch_manager.export_seed_files(resume_workspace, run_examples)
        else:
            checkpoint(
                "Saved pool state was found, but no active candidate workspace exists; "
                "falling back to accepted-version resume."
            )

    if not resumed_from_pool:
        checkpoint("Creating seed workspace copy", detail=True)
        if args.resume:
            current_workspace, resume_source = patch_manager.prepare_resume_workspace(examples)
            checkpoint(f"Resume source: {resume_source}", detail=True)
        else:
            current_workspace = patch_manager.prepare_initial_workspace(examples)
        run_examples = examples
        baseline_lines = evaluator.count_lines_of_code(current_workspace)

        seed_run_label = "resume_baseline" if args.resume else "baseline"
        checkpoint(f"Running {seed_run_label} in {current_workspace}")
        baseline_env = None if args.baseline_only else {"SFM_SAVE_MEDIA": "0"}
        if baseline_env:
            checkpoint(
                f"{seed_run_label}: media disabled because base will be calibrated before child generation",
                detail=True,
            )
        baseline_run = runner.run(
            current_workspace,
            seed_run_label,
            run_examples,
            PROJECT_DIR / "outputs",
            env_overrides=baseline_env,
        )
        checkpoint(f"Evaluating {seed_run_label} metrics", detail=True)
        baseline_metrics = evaluator.evaluate(baseline_run.to_dict(), current_workspace, baseline_lines)
        checkpoint(f"Baseline aggregate metrics: {summarize_metrics(baseline_metrics)}")
        if baseline_run.crashed or baseline_run.timed_out:
            checkpoint(
                f"Stopping safely: {seed_run_label} simulation failed before calibration "
                f"(crashed={baseline_run.crashed}, timed_out={baseline_run.timed_out})."
            )
            failure = baseline_metrics.get("simulation_failure", {})
            stderr_path = failure.get("stderr_path")
            if stderr_path:
                checkpoint(f"Failure log: {stderr_path}")
            checkpoint(
                "Fix the simulation environment or model error first; calibration only runs after "
                "a successful baseline."
            )
            print_resume_hint(args, 1, configured_max_iterations)
            return 1
        checkpoint(
            f"Baseline per-scenario metrics: {summarize_scenario_metrics(baseline_metrics)}",
            detail=True,
        )
        baseline_visual = {}
        if args.baseline_only:
            checkpoint("Analyzing baseline video/image outputs")
            baseline_visual = video_analyzer.analyze_outputs(Path(baseline_run.output_dir), current_workspace)
        current_metrics = baseline_metrics
        current_run = baseline_run
        current_visual = baseline_visual
        checkpoint("Computing baseline complexity snapshot", detail=True)
        current_complexity = compute_complexity_snapshot(current_workspace, config, "baseline")
        checkpoint(f"Baseline complexity: {summarize_complexity_snapshot(current_complexity)}")

        if args.baseline_only:
            terminal_print(json.dumps({"baseline_metrics": baseline_metrics, "visual": baseline_visual}, indent=2))
            return 0

        base_candidate = {
            "candidate_id": "base",
            "parent_id": None,
            "iteration_created": 0,
            "has_produced_child": False,
            "produced_child_count": 0,
            "status": "active",
            "workspace": str(current_workspace),
            "baseline_workspace": str(current_workspace),
            "baseline_metrics": merge_software_complexity(current_metrics, current_complexity),
            "baseline_complexity": current_complexity,
            "baseline_run_result": current_run.to_dict(),
            "metrics": merge_software_complexity(current_metrics, current_complexity),
            "complexity": current_complexity,
            "media_visual": current_visual,
            "media_run_result": current_run.to_dict() if args.baseline_only else None,
            "llm_candidate": {"name": "base"},
        }
        if args.skip_base_calibration:
            checkpoint(
                "Skipping base GA calibration; using current/default base parameters "
                "as already calibrated"
            )
            if config.get("calibration", {}).get("cleanup_after_success", False):
                checkpoint("Promoting skipped-calibration base workspace into model store", detail=True)
                original_base_workspace = base_candidate.get("workspace")
                promoted_base = promote_calibrated_workspace(base_candidate, Path(base_candidate["workspace"]), config)
                base_candidate["workspace"] = str(promoted_base)
                base_candidate["baseline_workspace"] = str(promoted_base)
                cleanup_disposable_workspace(original_base_workspace, config)
            base_candidate["base_calibration_skipped"] = True
            base_candidate["calibration_result"] = {
                "status": "skipped",
                "reason": "--skip_base_calibration was used.",
                "promoted_workspace": base_candidate.get("workspace"),
            }
            base_candidate["calibration_best_record"] = {
                "candidate_id": "base_default_parameters",
                "params": {},
                "metrics": base_candidate.get("metrics", {}),
                "workspace": base_candidate.get("workspace"),
                "reason": "Base parameter optimization skipped by command flag.",
            }
        else:
            checkpoint("Calibrating base before it can produce children")
            base_candidate = calibrate_child_candidate(
                base_candidate,
                config,
                runner,
                evaluator,
                run_examples,
                1,
            )
            if base_candidate.get("status") != "active":
                checkpoint(
                    "Stopping safely: base calibration failed before child generation "
                    f"({base_candidate.get('failure_reason', base_candidate.get('status'))})"
                )
                print_resume_hint(args, 1, configured_max_iterations)
                return 1
        media_label = "base MP4/PNG" if args.skip_base_calibration else "calibrated base MP4/PNG"
        checkpoint(f"Generating {media_label} for first OpenAI reasoning step")
        base_candidate = generate_media_for_candidate(
            base_candidate,
            runner,
            evaluator,
            video_analyzer,
            config,
            run_examples,
            baseline_lines,
        )
        try:
            bootstrap_autocred_if_needed(
                base_candidate,
                config,
                runner,
                run_examples,
                evaluator,
                video_analyzer,
                baseline_lines,
            )
        except Exception as exc:  # noqa: BLE001 - stop cleanly with persisted baseline context.
            checkpoint(f"Stopping safely: AutoCred bootstrap failed with {type(exc).__name__}: {exc}")
            print_resume_hint(args, 1, configured_max_iterations)
            return 1
        candidate_pool = [base_candidate]
        history.save_pool_state(
            make_pool_state(candidate_pool, 0, run_examples, baseline_lines, domination_sequence)
        )
        checkpoint(f"Saved baseline pool resume state: {history.pool_state_path}", detail=True)

    if resumed_from_pool and autocred_enabled(config) and not autocred_state_path(config).is_file():
        base_for_autocred = next(
            (candidate for candidate in candidate_pool if candidate.get("candidate_id") == "base"),
            None,
        )
        if base_for_autocred is None:
            base_for_autocred = next(
                (candidate for candidate in candidate_pool if candidate.get("status") == "active"),
                None,
            )
        if base_for_autocred is None:
            checkpoint("Stopping safely: AutoCred is enabled but no candidate is available to bootstrap the rubric.")
            print_resume_hint(args, 1, configured_max_iterations)
            return 1
        try:
            bootstrap_autocred_if_needed(
                base_for_autocred,
                config,
                runner,
                run_examples,
                evaluator,
                video_analyzer,
                baseline_lines,
            )
        except Exception as exc:  # noqa: BLE001 - stop cleanly on missing/invalid rubric.
            checkpoint(f"Stopping safely: AutoCred bootstrap failed with {type(exc).__name__}: {exc}")
            print_resume_hint(args, 1, configured_max_iterations)
            return 1

    start_iteration, max_iterations = iteration_bounds(args, config, history, active_pool_state)
    if start_iteration > max_iterations:
        checkpoint(
            f"No iterations to run: start_iteration={start_iteration} "
            f"is greater than max_iterations={max_iterations}."
        )
        print_resume_hint(args, start_iteration, start_iteration + int(config.get("max_iterations", 5)) - 1)
        return 0

    last_completed_iteration = int(
        (active_pool_state or {}).get("last_completed_iteration", start_iteration - 1) or 0
    )
    for iteration in range(start_iteration, max_iterations + 1):
        terminal_print(f"=== Iteration {iteration}/{max_iterations} ===")
        active_before = [candidate for candidate in candidate_pool if candidate.get("status") == "active"]
        checkpoint(f"Active pool before iteration: {len(active_before)} candidate(s)")
        allocations = allocate_child_budget(
            candidate_pool,
            MAX_CHILDREN_PER_ITERATION,
            iteration,
            int(config.get("calibration", {}).get("seed", 42)),
        )
        if not allocations:
            checkpoint("No active candidates remain; stopping pool refinement.")
            break
        ranked_active = [
            candidate
            for candidate in candidate_pool
            if candidate.get("status") == "active"
        ]
        ranked_active.sort(
            key=lambda candidate: (
                candidate.get("pool_topsis_rank") is None,
                candidate.get("pool_topsis_rank") or 999999,
                str(candidate.get("candidate_id", "")),
            )
        )
        checkpoint(
            "Active pool TOPSIS ranking: "
            + ", ".join(
                f"{candidate.get('candidate_id')}#rank={candidate.get('pool_topsis_rank', 'n/a')}"
                f"/score={candidate.get('pool_topsis_score', 'n/a')}"
                for candidate in ranked_active
            ),
            detail=True,
        )
        checkpoint(
            "Child allocation: "
            + ", ".join(
                f"{parent['candidate_id']}->{child_count}"
                for parent, child_count in allocations
            )
        )

        new_children: list[dict[str, Any]] = []
        diagnoses: list[dict[str, Any]] = []
        prepared_codex_jobs: list[dict[str, Any]] = []
        iteration_history_context = diagnosis_history_context(history, config)
        diagnosis_jobs = [
            {
                "parent": parent,
                "child_count": child_count,
                "iteration": iteration,
                "llm": llm,
                "history_context": iteration_history_context,
                "config": config,
                "run_examples": run_examples,
            }
            for parent, child_count in allocations
        ]
        diagnosis_workers = resolve_parallel_workers(
            config.get("diagnosis_parallel_workers", 1),
            len(diagnosis_jobs),
        )
        checkpoint(
            f"Running {len(diagnosis_jobs)} parent diagnosis request(s) "
            f"on {diagnosis_workers} worker(s)"
        )
        try:
            diagnosis_results = run_diagnosis_jobs(
                diagnosis_jobs,
                config.get("diagnosis_parallel_workers", 1),
            )
        except Exception as exc:  # noqa: BLE001 - stop cleanly with persisted context.
            checkpoint(f"Stopping safely: LLM diagnosis failed with {type(exc).__name__}: {exc}")
            print_resume_hint(args, iteration, max_iterations)
            return 1

        for diagnosis_result in diagnosis_results:
            parent = diagnosis_result["parent"]
            child_count = int(diagnosis_result["child_count"])
            parent_workspace = Path(diagnosis_result["parent_workspace"])
            parent_metrics = diagnosis_result["parent_metrics"]
            code_excerpt = diagnosis_result["code_excerpt"]
            diagnosis = diagnosis_result["diagnosis"]
            diagnoses.append(diagnosis)
            selected_children = select_child_candidates(diagnosis, child_count)
            checkpoint(
                f"Selected child option(s) for parent {parent['candidate_id']}: "
                + ", ".join(str(child.get("id")) for child in selected_children),
                detail=True,
            )
            produced_now = len(selected_children)
            parent["has_produced_child"] = True
            parent["produced_child_count"] = int(parent.get("produced_child_count", 0) or 0) + produced_now
            parent["last_produced_iteration"] = iteration
            checkpoint(
                f"Updated parent {parent['candidate_id']} production count: "
                f"+{produced_now}, total={parent['produced_child_count']}",
                detail=True,
            )

            for child_index, llm_child in enumerate(selected_children, start=1):
                child_id = child_candidate_id(iteration, parent, child_index)
                checkpoint(f"Creating child workspace {child_id}: {llm_child.get('name', 'unnamed child')}")
                child_workspace = patch_manager.create_candidate_workspace(
                    parent_workspace,
                    iteration,
                    run_examples,
                )
                checkpoint(f"{child_id}: workspace ready at {child_workspace}", detail=True)
                if not config.get("automatic_patch_approval", True):
                    terminal_print(json.dumps(llm_child, indent=2))
                    answer = timestamped_input(
                        f"Apply child {child_id} with Codex CLI? [y/N] "
                    ).strip().lower()
                    if answer != "y":
                        checkpoint(f"Skipping child {child_id} before patch application.")
                        continue

                edit_scope_baseline = patch_manager.capture_edit_scope(child_workspace)
                prepared_codex_jobs.append(
                    {
                        "child_id": child_id,
                        "parent": parent,
                        "iteration": iteration,
                        "child_index": child_index,
                        "llm_child": llm_child,
                        "diagnosis": diagnosis,
                        "child_workspace": child_workspace,
                        "edit_scope_baseline": edit_scope_baseline,
                        "parent_metrics": parent_metrics,
                        "code_excerpt": code_excerpt,
                        "failed_history": iteration_history_context,
                        "llm": llm,
                        "patch_manager": patch_manager,
                        "config": config,
                    }
                )

        codex_workers = resolve_parallel_workers(
            config.get("codex_cli", {}).get("parallel_workers", 1),
            len(prepared_codex_jobs),
        )
        checkpoint(
            f"Running {len(prepared_codex_jobs)} Codex CLI edit(s) "
            f"on {codex_workers} worker(s)"
        )
        if prepared_codex_jobs:
            try:
                new_children = run_codex_edit_jobs(
                    prepared_codex_jobs,
                    config.get("codex_cli", {}).get("parallel_workers", 1),
                )
            except Exception as exc:  # noqa: BLE001 - keep the resume command visible on edit failures.
                checkpoint(f"Stopping safely: Codex edit failed with {type(exc).__name__}: {exc}")
                print_resume_hint(args, iteration, max_iterations)
                return 1

        if not new_children:
            checkpoint("No child workspaces were created; stopping pool refinement.")
            print_resume_hint(args, iteration, max_iterations)
            return 1

        checkpoint(f"Running {len(new_children)} child candidate(s) metrics-only in parallel")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(len(new_children), MAX_CHILDREN_PER_ITERATION))) as executor:
            futures = {
                executor.submit(
                    run_child_metrics_only,
                    child,
                    runner,
                    evaluator,
                    config,
                    run_examples,
                    baseline_lines,
                ): child["candidate_id"]
                for child in new_children
            }
            completed_children = []
            for future in concurrent.futures.as_completed(futures):
                child_id = futures[future]
                checkpoint(f"{child_id}: metrics-only future completed; collecting result", detail=True)
                completed_children.append(future.result())
            new_children = completed_children

        valid_children = [child for child in new_children if child.get("status") == "active"]
        failed_after_initial = [child for child in new_children if child.get("status") != "active"]
        if failed_after_initial:
            checkpoint(
                "Children inactive after initial run: "
                + ", ".join(
                    f"{child.get('candidate_id')}({child.get('failure_reason', child.get('status'))})"
                    for child in failed_after_initial
                )
            )
        checkpoint(f"Calibrating {len(valid_children)} valid child candidate(s) in parallel")
        if valid_children:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(valid_children)) as executor:
                futures = {
                    executor.submit(
                        calibrate_child_candidate,
                        child,
                        config,
                        runner,
                        evaluator,
                        run_examples,
                        len(valid_children),
                    ): child["candidate_id"]
                    for child in valid_children
                }
                calibrated_by_id = {}
                for future in concurrent.futures.as_completed(futures):
                    child_id = futures[future]
                    checkpoint(f"{child_id}: calibration future completed; collecting result", detail=True)
                    calibrated = future.result()
                    calibrated_by_id[calibrated["candidate_id"]] = calibrated
            new_children = [
                calibrated_by_id.get(child["candidate_id"], child)
                if child.get("status") == "active"
                else child
                for child in new_children
            ]

        if autocred_enabled(config):
            plausibility_targets = [
                child for child in new_children if child.get("status") == "active"
            ]
            autocred_media_targets = [
                child
                for child in plausibility_targets
                if not (child.get("media_visual") and child.get("media_run_result"))
            ]
            if autocred_media_targets:
                media_workers = max(
                    1,
                    min(len(autocred_media_targets), MAX_PARALLEL_MEDIA_RUNS),
                )
                media_environment = divided_cpu_environment(media_workers)
                checkpoint(
                    "Preparing AutoCred media for "
                    f"{len(autocred_media_targets)} candidate(s) on "
                    f"{media_workers} worker(s)"
                )
                try:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=media_workers
                    ) as executor:
                        futures = {
                            executor.submit(
                                generate_media_for_candidate,
                                child,
                                runner,
                                evaluator,
                                video_analyzer,
                                config,
                                run_examples,
                                baseline_lines,
                                media_environment,
                            ): child["candidate_id"]
                            for child in autocred_media_targets
                        }
                        for future in concurrent.futures.as_completed(futures):
                            candidate_id = futures[future]
                            checkpoint(
                                f"{candidate_id}: AutoCred media future completed; "
                                "collecting result",
                                detail=True,
                            )
                            future.result()
                except Exception as exc:  # noqa: BLE001 - preserve the iteration for resume.
                    checkpoint(
                        "Stopping safely: AutoCred media preparation failed with "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print_resume_hint(args, iteration, max_iterations)
                    return 1

            autocred_workers = resolve_parallel_workers(
                config.get("autocred", {}).get("parallel_workers", 1),
                len(plausibility_targets),
            )
            checkpoint(
                f"Running AutoCred plausibility gate for "
                f"{len(plausibility_targets)} calibrated child candidate(s) "
                f"on {autocred_workers} worker(s)"
            )
            if plausibility_targets:
                try:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=autocred_workers
                    ) as executor:
                        futures = {
                            executor.submit(
                                run_autocred_plausibility_gate,
                                child,
                                config,
                                runner,
                                evaluator,
                                video_analyzer,
                                run_examples,
                                baseline_lines,
                            ): child
                            for child in plausibility_targets
                        }
                        for future in concurrent.futures.as_completed(futures):
                            child = futures[future]
                            future.result()
                            compact_candidate_outputs(child, config)
                except Exception as exc:  # noqa: BLE001 - stop cleanly with persisted context.
                    checkpoint(
                        "Stopping safely: AutoCred plausibility check failed with "
                        f"{type(exc).__name__}: {exc}"
                    )
                    print_resume_hint(args, iteration, max_iterations)
                    return 1
            try:
                new_children = retry_autocred_failed_children(
                    children=new_children,
                    candidate_pool=candidate_pool,
                    iteration=iteration,
                    llm=llm,
                    history=history,
                    patch_manager=patch_manager,
                    config=config,
                    runner=runner,
                    evaluator=evaluator,
                    video_analyzer=video_analyzer,
                    run_examples=run_examples,
                    baseline_lines=baseline_lines,
                    diagnoses=diagnoses,
                )
            except Exception:
                print_resume_hint(args, iteration, max_iterations)
                return 1

        candidate_pool.extend(new_children)
        checkpoint(f"Candidate pool size after adding children: {len(candidate_pool)}", detail=True)
        checkpoint("Applying hard domination filter across the full active pool", detail=True)
        before_domination_status = {
            candidate.get("candidate_id"): candidate.get("status")
            for candidate in candidate_pool
        }
        candidate_pool, domination_sequence = apply_hard_domination(candidate_pool, domination_sequence)
        newly_dominated = [
            candidate
            for candidate in candidate_pool
            if before_domination_status.get(candidate.get("candidate_id")) == "active"
            and candidate.get("status") == "dominated"
        ]
        if newly_dominated:
            checkpoint(
                "Newly dominated candidate(s): "
                + ", ".join(
                    f"{candidate.get('candidate_id')} by {candidate.get('dominated_by')} "
                    f"(order {candidate.get('dominated_order')})"
                    for candidate in newly_dominated
                )
            )
        else:
            checkpoint("No newly dominated candidates in this iteration", detail=True)
        active_after_filter = [
            candidate for candidate in candidate_pool if candidate.get("status") == "active"
        ]
        checkpoint(f"Active pool after domination filter: {len(active_after_filter)} candidate(s)")

        media_targets = [
            candidate
            for candidate in active_after_filter
            if not (candidate.get("media_visual") and candidate.get("media_run_result"))
        ]
        checkpoint(
            f"Generating MP4/PNG only for {len(media_targets)} active candidate(s) without media"
        )
        if media_targets:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, min(len(media_targets), MAX_PARALLEL_MEDIA_RUNS))
            ) as executor:
                futures = {
                    executor.submit(
                        generate_media_for_candidate,
                        candidate,
                        runner,
                        evaluator,
                        video_analyzer,
                        config,
                        run_examples,
                        baseline_lines,
                    ): candidate["candidate_id"]
                    for candidate in media_targets
                }
                for future in concurrent.futures.as_completed(futures):
                    candidate_id = futures[future]
                    checkpoint(f"{candidate_id}: media future completed; collecting result", detail=True)
                    future.result()

        archived_children: list[dict[str, Any]] = []
        literature_mappings = collect_literature_mappings(diagnoses, new_children)
        for child in new_children:
            accepted = child.get("status") == "active"
            archive_workspace = Path(child.get("workspace", child["parent_workspace"]))
            checkpoint(
                f"{child.get('candidate_id')}: archiving as "
                f"{'accepted' if accepted else 'rejected'} from {archive_workspace}",
                detail=True,
            )
            archive = patch_manager.archive_version(
                archive_workspace,
                accepted=accepted,
                iteration=iteration,
                metadata={
                    "mode": "pool_refinement",
                    "accepted": accepted,
                    "candidate_id": child.get("candidate_id"),
                    "parent_id": child.get("parent_id"),
                    "status": child.get("status"),
                    "dominated_by": child.get("dominated_by"),
                    "failure_reason": child.get("failure_reason"),
                    "metrics": child.get("metrics"),
                    "complexity": child.get("complexity"),
                    "calibration_result": child.get("calibration_result"),
                    "calibration_best_record": child.get("calibration_best_record"),
                    "autocred_plausibility": child.get("autocred_plausibility"),
                    "autocred_run_result": child.get("autocred_run_result"),
                    "replaces_candidate_id": child.get("replaces_candidate_id"),
                    "replacement_attempt": child.get("replacement_attempt"),
                    "validation": child.get("validation"),
                    "llm_candidate": child.get("llm_candidate"),
                    "literature_basis": child.get("llm_candidate", {}).get("literature_basis", []),
                },
            )
            child["archive_path"] = str(archive)
            checkpoint(f"{child.get('candidate_id')}: archive written to {archive}", detail=True)
            if not accepted:
                cleanup_disposable_workspace(child.get("workspace"), config)
                delete_candidate_outputs(child, config)
                child["workspace_cleaned_after_archive"] = True
            archived_children.append(
                {
                    "candidate_id": child.get("candidate_id"),
                    "status": child.get("status"),
                    "archive_path": str(archive),
                }
            )

        checkpoint("Writing iteration entry to history report", detail=True)
        history.append(
            {
                "mode": "pool_refinement",
                "iteration": iteration,
                "scenario": "all_examples",
                "examples": run_examples,
                "decision": "pool_updated",
                "llm_diagnosis": " | ".join(diagnosis.get("diagnosis", "") for diagnosis in diagnoses),
                "pool_before": [pool_candidate_summary(candidate) for candidate in active_before],
                "children_created": [pool_candidate_summary(child) for child in new_children],
                "pool_after": pool_summary(candidate_pool),
                "archived_children": archived_children,
                "literature_mappings": literature_mappings,
                "notes": (
                    f"Created {len(new_children)} child candidate(s), calibrated "
                    f"{len(valid_children)}, active survivors now {len(active_after_filter)}."
                ),
            }
        )
        history.save_pool_state(
            make_pool_state(candidate_pool, iteration, run_examples, baseline_lines, domination_sequence)
        )
        last_completed_iteration = iteration
        checkpoint(f"Saved pool resume state: {history.pool_state_path}", detail=True)

        terminal_print(
            f"Pool iteration {iteration}: created {len(new_children)} child candidate(s), "
            f"active survivors {len(active_after_filter)}."
        )

    graph_result = write_iteration_graph(candidate_pool, PROJECT_DIR / "history")
    checkpoint(f"Candidate graph JSON: {graph_result.get('json_path')}", detail=True)
    if graph_result.get("png_path"):
        checkpoint(f"Candidate graph PNG: {graph_result.get('png_path')}", detail=True)
    elif graph_result.get("error"):
        checkpoint(f"Candidate graph PNG not written: {graph_result.get('error')}")
    checkpoint(f"Done. Report: {PROJECT_DIR / 'history' / 'report.md'}")
    completed_batch_size = max(1, max_iterations - start_iteration + 1)
    next_iteration = last_completed_iteration + 1
    print_resume_hint(args, next_iteration, last_completed_iteration + completed_batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

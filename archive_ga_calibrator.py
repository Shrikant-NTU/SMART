from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from evaluator import Evaluator
from parameter_patch_manager import ParameterPatchManager
from parameter_space import (
    CALIBRATION_MANIFEST_NAME,
    load_parameter_specs,
    update_manifest_defaults,
)
from simulation_runner import SimulationRunner


CALIBRATION_SCRIPT_NAME = "calibrate_example10_ga.py"
GA_EXAMPLES = [10]
AGENT_METRIC_KEYS = [
    "goal_reach_rate",
    "average_completion_time",
    "completion_time",
    "stuck_pedestrian_count",
    "unique_overlap_count",
    "overlap_episode_count",
    "total_overlap_duration_seconds",
    "average_overlap_duration_seconds",
    "average_overlap_duration_per_pair_seconds",
    "overlap_count",
    "average_speed",
    "total_distance_traveled",
    "average_distance_traveled",
    "execution_time_seconds",
    "standard_cyclomatic_complexity",
    "halstead_effort",
]


class ArchiveGACalibrator:
    """Runs the agent-owned example-10 archive GA implementation."""

    def __init__(
        self,
        config: dict[str, Any],
        project_dir: Path,
        runner: SimulationRunner,
        evaluator: Evaluator,
        checkpoint: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.project_dir = project_dir
        self.runner = runner
        self.evaluator = evaluator
        self.checkpoint = checkpoint or (lambda message: None)
        self.calibration_config = config.get("calibration", {})

    def run(
        self,
        seed_workspace: Path,
        examples: list[int] | None,
        concurrent_model_count: int | None = None,
    ) -> dict[str, Any]:
        requested_examples = sorted(set(examples or GA_EXAMPLES))
        if requested_examples != GA_EXAMPLES:
            self.checkpoint(
                "Archive GA calibration is defined for example10; "
                f"calibrating example10 only (requested={requested_examples})."
            )

        script_path = self._calibration_script()
        run_token = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = self.project_dir / "calibration" / f"archive_ga_example10_{run_token}"
        module_name = f"_sfm_archive_ga_{uuid.uuid4().hex}"
        try:
            module = self._load_module(script_path, module_name)
            self._configure_module(
                module,
                seed_workspace=seed_workspace,
                run_token=run_token,
                concurrent_model_count=concurrent_model_count,
            )
            self.checkpoint(f"Archive GA source: {script_path}")
            self.checkpoint(f"Archive GA output: {run_dir}")
            exit_code = module.main()
            if exit_code not in (None, 0):
                return self._stopped_result(run_dir, f"GA exited with status {exit_code}.")
            return self._normalize_completed_run(run_dir, seed_workspace)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
            if code != 0:
                return self._stopped_result(run_dir, str(exc.code or "GA stopped."))
        except Exception as exc:  # noqa: BLE001 - preserve partial GA artifacts for diagnosis.
            run_dir.mkdir(parents=True, exist_ok=True)
            error = f"{type(exc).__name__}: {exc}"
            (run_dir / "agent_calibration_error.json").write_text(
                json.dumps({"status": "failed", "error": error}, indent=2),
                encoding="utf-8",
            )
            self.checkpoint(f"Archive GA failed: {error}")
            return self._stopped_result(run_dir, error)
        finally:
            sys.modules.pop(module_name, None)

    def _calibration_script(self) -> Path:
        path = self.project_dir / CALIBRATION_SCRIPT_NAME
        if not path.is_file():
            raise FileNotFoundError(f"Agent calibration script was not found: {path}")
        return path

    @staticmethod
    def _load_module(script_path: Path, module_name: str) -> ModuleType:
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load calibration module from {script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def _configure_module(
        self,
        module: ModuleType,
        seed_workspace: Path,
        run_token: str,
        concurrent_model_count: int | None,
    ) -> None:
        calibration = self.calibration_config
        module.SCRIPT_DIR = self.project_dir
        module.now_id = lambda: run_token
        module.parse_args = lambda: argparse.Namespace(
            save_best_media=bool(calibration.get("save_media", False))
        )
        module.copy_source_workspace = lambda destination: self._copy_candidate_source(
            seed_workspace, destination
        )
        module.run_example = self._run_example
        module.TERMINAL_VERBOSITY = str(
            self.config.get("terminal_verbosity", "concise")
        ).lower()
        module.TERMINAL_GENERATION_INTERVAL = max(
            1,
            int(self.config.get("terminal_ga_generation_interval", 5)),
        )
        module.TIMEOUT_SECONDS = int(self.config.get("timeout_seconds", module.TIMEOUT_SECONDS))

        parameter_specs = load_parameter_specs(self.config, seed_workspace)
        ParameterPatchManager().validate_targets(
            seed_workspace,
            parameter_specs,
            require_default_match=True,
        )
        module.PARAMETER_SPECS = {
            spec.name: spec.ga_spec()
            for spec in parameter_specs
        }
        module.PARAMETER_MUTATION_SIZES = {
            spec.name: spec.mutation_size
            for spec in parameter_specs
        }
        parameter_patcher = ParameterPatchManager()
        module.patch_parameter_file = lambda workspace, values: parameter_patcher.apply(
            workspace,
            parameter_specs,
            values,
        )

        module.INITIAL_POPULATION_SIZE = int(
            calibration.get("initial_population_size", module.INITIAL_POPULATION_SIZE)
        )
        module.WORKING_POPULATION_SIZE = int(
            calibration.get("working_population_size", module.WORKING_POPULATION_SIZE)
        )
        module.GENERATIONS = int(calibration.get("generations", module.GENERATIONS))
        module.CROSSOVER_PROBABILITY = float(
            calibration.get("crossover_probability", module.CROSSOVER_PROBABILITY)
        )
        module.MUTATION_PROBABILITY = float(
            calibration.get("mutation_probability", module.MUTATION_PROBABILITY)
        )
        module.RANDOM_SEED = int(calibration.get("seed", module.RANDOM_SEED))
        module.CALIBRATION_REPEAT_COUNT = int(
            calibration.get("repeat_count", module.CALIBRATION_REPEAT_COUNT)
        )
        repeat_seeds = calibration.get("repeat_seeds", module.CALIBRATION_REPEAT_SEEDS)
        module.CALIBRATION_REPEAT_SEEDS = [int(seed) for seed in repeat_seeds]
        module.MAX_AVERAGE_OVERLAP_DURATION_SECONDS = float(
            calibration.get(
                "max_average_overlap_duration_seconds",
                module.MAX_AVERAGE_OVERLAP_DURATION_SECONDS,
            )
        )
        module.MAX_FEASIBILITY_EVALUATION_MULTIPLIER = int(
            calibration.get(
                "max_feasibility_evaluation_multiplier",
                module.MAX_FEASIBILITY_EVALUATION_MULTIPLIER,
            )
        )
        module.MAX_FEASIBILITY_REPLACEMENTS_TOTAL = int(
            calibration.get(
                "max_feasibility_replacements_total",
                module.MAX_FEASIBILITY_REPLACEMENTS_TOTAL,
            )
        )
        module.PARALLEL_WORKERS = self._parallel_workers(
            calibration.get("parallel_workers", module.PARALLEL_WORKERS),
            concurrent_model_count,
        )

        topsis_weights = calibration.get("topsis_weights", {})
        for metric, value in topsis_weights.items():
            if metric in module.TOPSIS_WEIGHTS:
                module.TOPSIS_WEIGHTS[metric] = float(value)

    @staticmethod
    def _parallel_workers(configured: Any, concurrent_model_count: int | None) -> int | str:
        if configured != "all":
            return max(1, int(configured))
        model_count = max(1, int(concurrent_model_count or 1))
        if model_count == 1:
            return "all"
        return max(1, (os.cpu_count() or 1) // model_count)

    @staticmethod
    def _copy_candidate_source(seed_workspace: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=False)
        jsfm_source = seed_workspace / "jsfm"
        example_path = seed_workspace / "example10.py"
        if not example_path.is_file():
            raise FileNotFoundError(f"Example 10 was not found: {example_path}")
        if not jsfm_source.is_dir():
            raise FileNotFoundError(f"JSFM package was not found: {jsfm_source}")

        for source_path in sorted(seed_workspace.glob("example*.py")):
            shutil.copy2(source_path, destination / source_path.name)
        runner_script = seed_workspace / "run_examples.py"
        if runner_script.is_file():
            shutil.copy2(runner_script, destination / runner_script.name)
        manifest = seed_workspace / CALIBRATION_MANIFEST_NAME
        if manifest.is_file():
            shutil.copy2(manifest, destination / manifest.name)
        shutil.copytree(
            jsfm_source,
            destination / "jsfm",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

    def _run_example(
        self,
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
            [str(self.runner.python_executable), "example10.py"],
            cwd=workspace,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(self.config.get("timeout_seconds", 3600)),
            check=False,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "runtime_seconds": time.perf_counter() - start,
        }

    def _normalize_completed_run(
        self,
        run_dir: Path,
        seed_workspace: Path,
    ) -> dict[str, Any]:
        best_path = run_dir / "best_calibration.json"
        payload = json.loads(best_path.read_text(encoding="utf-8"))
        native_best_path = run_dir / "ga_best_calibration.json"
        native_best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        best_record = dict(payload["best_candidate"])
        best_workspace = Path(best_record["workspace"])
        update_manifest_defaults(
            best_workspace,
            self.config,
            best_record.get("parameters", {}),
        )
        source_manifest = seed_workspace / CALIBRATION_MANIFEST_NAME
        if source_manifest.is_file():
            shutil.copy2(source_manifest, run_dir / CALIBRATION_MANIFEST_NAME)
        metrics = self._agent_metrics(best_workspace, seed_workspace)
        selected_metrics = {
            key: metrics.get(key)
            for key in AGENT_METRIC_KEYS
            if key in metrics
        }
        final_score = best_record.get(
            "final_topsis_score", best_record.get("current_topsis_score")
        )
        best_record.update(
            {
                "metrics": metrics,
                "selected_metrics": selected_metrics,
                "topsis_score": final_score,
                "rank": 1,
                "selected_by_topsis": True,
                "selection_rule": "cumulative archive TOPSIS over feasible GA candidates",
                "run_result": self._agent_run_result(best_record, best_workspace),
            }
        )
        best_path.write_text(json.dumps(best_record, indent=2), encoding="utf-8")

        records_path = run_dir / "all_records.json"
        all_records = json.loads(records_path.read_text(encoding="utf-8"))
        for record in all_records:
            record["selected_metrics"] = dict(record.get("topsis_metrics", {}))
            record["topsis_score"] = record.get(
                "final_topsis_score", record.get("current_topsis_score")
            )
            record["selected_by_topsis"] = record.get("candidate_id") == best_record.get(
                "candidate_id"
            )
            record["selection_rule"] = (
                "cumulative archive TOPSIS over feasible GA candidates"
                if record.get("is_feasible")
                else "excluded by feasibility constraint"
            )
            record["run_result"] = self._agent_run_result(
                record, Path(record["workspace"])
            )
        records_path.write_text(json.dumps(all_records, indent=2), encoding="utf-8")

        candidate_pool = sorted(
            (record for record in all_records if record.get("is_feasible")),
            key=lambda record: self._finite_score(record.get("topsis_score")),
            reverse=True,
        )
        candidate_pool_path = run_dir / "candidate_pool.json"
        candidate_pool_path.write_text(
            json.dumps(candidate_pool, indent=2), encoding="utf-8"
        )

        result = {
            "run_dir": str(run_dir),
            "evaluated_candidates": len(all_records),
            "feasible_candidates": len(candidate_pool),
            "removed_candidates": len(all_records) - len(candidate_pool),
            "best_calibration_candidate_id": best_record.get("candidate_id"),
            "best_calibration_path": str(best_path),
            "native_ga_best_calibration_path": str(native_best_path),
            "candidate_pool_path": str(candidate_pool_path),
            "removed_candidates_path": str(run_dir / "removed_candidates.json"),
            "report_path": str(run_dir / "report.md"),
            "plots_dir": str(run_dir / "plots"),
            "calibration_manifest_path": str(best_workspace / CALIBRATION_MANIFEST_NAME),
            "optimizer": "archive_GA_cumulative_pymcdm_TOPSIS",
            "examples": GA_EXAMPLES,
        }
        (run_dir / "agent_integration.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    def _agent_metrics(self, workspace: Path, seed_workspace: Path) -> dict[str, Any]:
        metrics_path = workspace / ".images" / "example10_metrics.json"
        raw_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics = dict(raw_metrics)
        scenario_metrics = raw_metrics.get("scenario_metrics")
        if isinstance(scenario_metrics, list) and scenario_metrics:
            first = scenario_metrics[0]
            if isinstance(first, dict):
                metrics.update(first)
        if metrics.get("average_completion_time") is None:
            metrics["average_completion_time"] = metrics.get("completion_time")
        metrics.update(self._software_complexity(workspace))
        metrics["lines_of_code"] = self.evaluator.count_lines_of_code(workspace)
        metrics["lines_added_vs_baseline"] = (
            metrics["lines_of_code"] - self.evaluator.count_lines_of_code(seed_workspace)
        )
        return metrics

    def _software_complexity(self, workspace: Path) -> dict[str, float]:
        try:
            from software_complexity_analyzer import analyze_software_complexity

            software_config = self.config.get("software_complexity", {})
            start_path = str(
                software_config.get(
                    "start_path",
                    self.config.get("model_file_name", "jsfm/sfm.py"),
                )
            )
            start_function = str(
                software_config.get("start_function", "single_update_active")
            ).strip()
            if not (workspace / start_path).is_file():
                return {}
            report = analyze_software_complexity(
                workspace,
                start_path,
                start_function,
            )
            resolved_scope = report["resolved_project_scope"]
            return {
                "standard_cyclomatic_complexity": float(
                    resolved_scope["standard_cyclomatic_complexity"]
                ),
                "halstead_effort": float(resolved_scope["halstead"]["effort"]),
            }
        except Exception as exc:  # noqa: BLE001 - calibration metrics remain usable.
            self.checkpoint(
                f"Software complexity unavailable for {workspace.name}: "
                f"{type(exc).__name__}: {exc}"
            )
            return {
                "standard_cyclomatic_complexity": math.inf,
                "halstead_effort": math.inf,
            }

    @staticmethod
    def _agent_run_result(record: dict[str, Any], workspace: Path) -> dict[str, Any]:
        native = dict(record.get("run_result", {}))
        returncode = int(native.get("returncode", 1))
        native.update(
            {
                "workspace": str(workspace),
                "output_dir": str(workspace / ".images"),
                "examples": GA_EXAMPLES,
                "timed_out": "timed out" in str(record.get("error", "")).lower(),
                "crashed": returncode != 0,
            }
        )
        return native

    def _stopped_result(self, run_dir: Path, reason: str) -> dict[str, Any]:
        self.checkpoint(f"Archive GA stopped without a selected candidate: {reason}")
        all_records_path = run_dir / "all_records.json"
        removed_path = run_dir / "removed_candidates.json"
        stopped_path = run_dir / "calibration_stopped.json"
        return {
            "run_dir": str(run_dir),
            "evaluated_candidates": self._json_list_length(all_records_path),
            "feasible_candidates": 0,
            "removed_candidates": self._json_list_length(removed_path),
            "best_calibration_candidate_id": None,
            "best_calibration_path": None,
            "candidate_pool_path": None,
            "removed_candidates_path": str(removed_path),
            "report_path": str(stopped_path if stopped_path.exists() else run_dir),
            "optimizer": "archive_GA_cumulative_pymcdm_TOPSIS",
            "examples": GA_EXAMPLES,
            "status": "stopped",
            "reason": reason,
        }

    @staticmethod
    def _json_list_length(path: Path) -> int:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return len(value) if isinstance(value, list) else 0
        except (OSError, json.JSONDecodeError):
            return 0

    @staticmethod
    def _finite_score(value: Any) -> float:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return -math.inf
        return score if math.isfinite(score) else -math.inf

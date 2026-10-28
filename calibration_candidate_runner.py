from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from evaluator import Evaluator
from parameter_patch_manager import ParameterPatchManager
from parameter_space import CALIBRATION_MANIFEST_NAME, load_parameter_specs
from patch_manager import PatchManager
from simulation_runner import SimulationRunner


class CalibrationCandidateRunner:
    def __init__(
        self,
        config: dict[str, Any],
        project_dir: Path,
        patch_manager: PatchManager,
        runner: SimulationRunner,
        evaluator: Evaluator,
        checkpoint: Callable[[str], None] | None = None,
    ):
        self.config = config
        self.project_dir = project_dir
        self.patch_manager = patch_manager
        self.runner = runner
        self.evaluator = evaluator
        self.checkpoint = checkpoint or (lambda message: None)
        self.parameter_patch_manager = ParameterPatchManager()

    def run(
        self,
        candidate_number: int,
        examples: list[int] | None,
        calibration_run_dir: Path | None = None,
        save_media: bool = True,
    ) -> dict[str, Any]:
        run_dir = calibration_run_dir or self.latest_calibration_run_dir()
        candidate_pool = self._load_candidate_pool(run_dir)
        candidate = self._candidate_by_number(candidate_pool, candidate_number)
        candidate_examples = examples or candidate.get("run_result", {}).get("examples") or None

        seed_source = Path(str(candidate.get("workspace", "")))
        if not seed_source.is_dir():
            raise FileNotFoundError(
                "The selected candidate's model workspace is unavailable; "
                f"cannot safely replay its candidate-specific mechanism: {seed_source}"
            )
        workspace = self._candidate_workspace(candidate_number)
        self.patch_manager.copy_candidate_files(
            seed_source,
            workspace,
            candidate_examples,
        )
        specs = load_parameter_specs(self.config, workspace)
        self.parameter_patch_manager.apply(workspace, specs, candidate.get("parameters", {}))

        self.checkpoint(f"Replaying calibration candidate {candidate_number} from {run_dir}")
        self.checkpoint(f"Seed source: {seed_source}")
        self.checkpoint(f"Candidate parameters: {json.dumps(candidate.get('parameters', {}), sort_keys=True)}")

        env_overrides = None if save_media else {"SFM_SAVE_MEDIA": "0"}
        run_result = self.runner.run(
            workspace,
            f"replay_calibration_candidate_{candidate_number:03d}",
            candidate_examples,
            self.project_dir / "outputs",
            env_overrides=env_overrides,
        )
        metrics = self.evaluator.evaluate(
            run_result.to_dict(),
            workspace,
            self.evaluator.count_lines_of_code(workspace),
        )
        replay_summary = {
            "candidate_number": candidate_number,
            "candidate_id": candidate.get("candidate_id"),
            "source_calibration_run": str(run_dir),
            "workspace": str(workspace),
            "output_dir": str(run_result.output_dir),
            "examples": run_result.examples,
            "parameters": candidate.get("parameters", {}),
            "calibration_manifest": str(workspace / CALIBRATION_MANIFEST_NAME),
            "metrics": metrics,
            "save_media": save_media,
        }
        (Path(run_result.output_dir) / "replayed_calibration_candidate.json").write_text(
            json.dumps(replay_summary, indent=2),
            encoding="utf-8",
        )
        return replay_summary

    def latest_calibration_run_dir(self) -> Path:
        calibration_root = self.project_dir / "calibration"
        if not calibration_root.exists():
            raise FileNotFoundError("No calibration directory found.")
        candidates = [
            path
            for path in calibration_root.iterdir()
            if path.is_dir()
            and (
                (path / "candidate_pool.json").exists()
                or (path / "pareto_front.json").exists()
            )
        ]
        if not candidates:
            raise FileNotFoundError("No calibration run with candidate_pool.json was found.")
        return max(candidates, key=lambda path: path.stat().st_mtime)

    @staticmethod
    def _load_candidate_pool(run_dir: Path) -> list[dict[str, Any]]:
        path = run_dir / "candidate_pool.json"
        if not path.exists():
            path = run_dir / "pareto_front.json"
        if not path.exists():
            raise FileNotFoundError(f"Calibration candidate pool not found: {run_dir}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Calibration candidate pool must be a list: {path}")
        return data

    @staticmethod
    def _candidate_by_number(candidate_pool: list[dict[str, Any]], candidate_number: int) -> dict[str, Any]:
        if candidate_number < 1 or candidate_number > len(candidate_pool):
            raise IndexError(
                f"Candidate number {candidate_number} is out of range. "
                f"Candidate pool has {len(candidate_pool)} candidate(s)."
            )
        return candidate_pool[candidate_number - 1]

    def _candidate_workspace(self, candidate_number: int) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.project_dir / "working_model" / f"calibration_candidate_{candidate_number:03d}_{timestamp}"

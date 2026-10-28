from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


EXAMPLE_PATTERN = re.compile(r"example(\d+)\.py$")


@dataclass
class ScriptRun:
    example_number: int
    script_name: str
    returncode: int
    runtime_seconds: float
    timed_out: bool
    stdout_path: str
    stderr_path: str
    error: str = ""


@dataclass
class SimulationRunResult:
    run_id: str
    workspace: str
    output_dir: str
    examples: list[int]
    returncode: int
    runtime_seconds: float
    timed_out: bool
    crashed: bool
    script_runs: list[ScriptRun]
    copied_outputs: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["script_runs"] = [asdict(item) for item in self.script_runs]
        return data


class SimulationRunner:
    """Runs copied SFM workspaces and archives produced files per iteration."""

    def __init__(self, config: dict[str, Any], project_dir: Path):
        self.config = config
        self.project_dir = project_dir
        self.timeout_seconds = int(config.get("timeout_seconds", 3600))
        self.python_executable = self._resolve_path(
            config.get("python_executable", "python")
        )
        self.output_dir_name = config.get("output_dir_name", ".images")

    def discover_examples(self, workspace: Path) -> dict[int, Path]:
        examples: dict[int, Path] = {}
        for path in workspace.glob(self.config.get("example_glob", "example*.py")):
            match = EXAMPLE_PATTERN.fullmatch(path.name)
            if match:
                examples[int(match.group(1))] = path
        return dict(sorted(examples.items()))

    def select_examples(self, workspace: Path, requested: list[int] | None) -> list[int]:
        discovered = self.discover_examples(workspace)
        if not requested:
            return list(discovered)

        missing = [number for number in requested if number not in discovered]
        if missing and self.config.get("missing_example_policy", "skip") == "error":
            raise FileNotFoundError(f"Missing example script(s): {missing}")
        selected = [number for number in requested if number in discovered]
        if requested and not selected:
            raise FileNotFoundError(
                f"No requested example scripts were found. Requested={requested}, missing={missing}"
            )
        return selected

    def run(
        self,
        workspace: Path,
        run_id: str,
        examples: list[int] | None,
        output_root: Path,
        env_overrides: dict[str, str] | None = None,
    ) -> SimulationRunResult:
        workspace = workspace.resolve()
        output_dir = self._unique_output_dir(output_root, run_id)
        output_dir.mkdir(parents=True, exist_ok=False)
        logs_dir = output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        selected = self.select_examples(workspace, examples)
        script_runs: list[ScriptRun] = []
        copied_outputs: list[str] = []
        run_start = time.perf_counter()
        crashed = False
        timed_out = False

        environment = os.environ.copy()
        if self.config.get("headless_matplotlib", True):
            environment["MPLBACKEND"] = "Agg"
        if env_overrides:
            environment.update(env_overrides)

        discovered = self.discover_examples(workspace)
        for position, number in enumerate(selected, start=1):
            script = discovered[number]
            stdout_path = logs_dir / f"example{number}_stdout.txt"
            stderr_path = logs_dir / f"example{number}_stderr.txt"
            start = time.perf_counter()
            error = ""
            returncode = 1
            script_timed_out = False

            try:
                result = subprocess.run(
                    [str(self.python_executable), str(script.name)],
                    cwd=workspace,
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                    check=False,
                )
                returncode = result.returncode
                stdout_path.write_text(result.stdout or "", encoding="utf-8", errors="replace")
                stderr_path.write_text(result.stderr or "", encoding="utf-8", errors="replace")
            except subprocess.TimeoutExpired as exc:
                script_timed_out = True
                timed_out = True
                returncode = 124
                error = f"Timed out after {self.timeout_seconds} seconds"
                stdout_path.write_text(exc.stdout or "", encoding="utf-8", errors="replace")
                stderr_path.write_text(exc.stderr or error, encoding="utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001 - persisted for diagnosis
                error = repr(exc)
                stderr_path.write_text(error, encoding="utf-8", errors="replace")
                stdout_path.write_text("", encoding="utf-8")

            elapsed = time.perf_counter() - start
            if returncode != 0:
                crashed = True

            script_runs.append(
                ScriptRun(
                    example_number=number,
                    script_name=script.name,
                    returncode=returncode,
                    runtime_seconds=elapsed,
                    timed_out=script_timed_out,
                    stdout_path=str(stdout_path),
                    stderr_path=str(stderr_path),
                    error=error,
                )
            )

            if returncode != 0:
                break

        runtime_seconds = time.perf_counter() - run_start
        images_dir = workspace / self.output_dir_name
        archived_images = output_dir / "images"
        if images_dir.exists():
            archived_images.mkdir(parents=True, exist_ok=True)
            for path in images_dir.iterdir():
                if path.is_file():
                    destination = archived_images / path.name
                    shutil.copy2(path, destination)
                    copied_outputs.append(str(destination))

        result = SimulationRunResult(
            run_id=run_id,
            workspace=str(workspace),
            output_dir=str(output_dir),
            examples=selected,
            returncode=0 if not crashed and not timed_out else 1,
            runtime_seconds=runtime_seconds,
            timed_out=timed_out,
            crashed=crashed,
            script_runs=script_runs,
            copied_outputs=copied_outputs,
        )
        (output_dir / "run_result.json").write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        return result

    def _resolve_path(self, value: str) -> Path:
        if value == "python":
            return Path(sys.executable)
        path = Path(value)
        if path.is_absolute():
            return path
        executable = shutil.which(value)
        if executable:
            return Path(executable).resolve()
        return (self.project_dir / path).resolve()

    @staticmethod
    def _unique_output_dir(output_root: Path, run_id: str) -> Path:
        output_root.mkdir(parents=True, exist_ok=True)
        candidate = output_root / run_id
        if not candidate.exists():
            return candidate
        index = 1
        while True:
            candidate = output_root / f"{run_id}_{index:03d}"
            if not candidate.exists():
                return candidate
            index += 1

from __future__ import annotations

import json
import os
import base64
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from openai_key import resolve_openai_api_key
from prompts import (
    CODEX_EDIT_PROMPT_TEMPLATE,
    DIAGNOSIS_PROMPT_TEMPLATE,
    JSON_REPAIR_PROMPT_TEMPLATE,
)


class LLMClient:
    """Routes reasoning through the configured backend and edits through Codex CLI."""

    def __init__(self, config: dict[str, Any], project_dir: Path, history_manager: Any):
        self.config = config
        self.project_dir = project_dir
        self.history = history_manager
        self.reasoning_backend = config.get(
            "reasoning_backend",
            config.get("llm_backend", "codex_cli"),
        )
        self.code_edit_backend = config.get("code_edit_backend", "codex_cli")
        self.terminal_verbosity = str(config.get("terminal_verbosity", "concise")).lower()
        self.progress_interval_seconds = max(
            30,
            int(config.get("terminal_progress_interval_seconds", 120)),
        )

    def diagnose(
        self,
        iteration: int,
        workspace: Path,
        examples: list[int],
        metrics: dict[str, Any],
        history: list[dict[str, Any]],
        visual_summary: dict[str, Any],
        code_excerpt: str,
        max_children: int = 2,
        parent_state: dict[str, Any] | None = None,
        image_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(
            max_children=max_children,
            parent_candidate_id=(parent_state or {}).get("candidate_id", "base"),
            parent_state_json=json.dumps(parent_state or {}, indent=2),
            examples=examples,
            metrics_json=json.dumps(metrics, indent=2),
            history_json=json.dumps(history, indent=2),
            visual_summary=json.dumps(visual_summary, indent=2),
            code_excerpt=code_excerpt,
            max_new_parameters=int(
                self.config.get("calibration", {}).get("max_new_parameters_per_candidate", 3)
            ),
        )
        response = self._call_text_model(
            prompt,
            workspace,
            read_only=True,
            image_paths=image_paths or [],
        )
        self.history.save_llm_exchange(iteration, "diagnosis", prompt, response)

        try:
            return self._parse_candidate_json(response, expected_count=max_children)
        except ValueError:
            repair_prompt = JSON_REPAIR_PROMPT_TEMPLATE.format(
                schema_description=(
                    "diagnosis string, candidates list with exactly "
                    f"{max_children} items, recommended_candidate_id integer"
                ),
                bad_response=response,
            )
            repaired = self._call_text_model(repair_prompt, workspace, read_only=True)
            self.history.save_llm_exchange(iteration, "diagnosis_repair", repair_prompt, repaired)
            return self._parse_candidate_json(repaired, expected_count=max_children)

    def apply_with_codex(
        self,
        iteration: int,
        candidate_workspace: Path,
        candidate: dict[str, Any],
        metrics: dict[str, Any],
        failed_history: list[dict[str, Any]],
        code_excerpt: str,
    ) -> str:
        if self.code_edit_backend != "codex_cli":
            raise ValueError("Only code_edit_backend: codex_cli is supported for code edits.")
        active_parameter_policy = self._active_parameter_policy_text()
        prompt = CODEX_EDIT_PROMPT_TEMPLATE.format(
            candidate_json=json.dumps(candidate, indent=2),
            metrics_json=json.dumps(metrics, indent=2),
            failed_history_json=json.dumps(failed_history, indent=2),
            code_excerpt=code_excerpt,
            max_new_parameters=int(
                self.config.get("calibration", {}).get("max_new_parameters_per_candidate", 3)
            ),
            active_parameter_policy=active_parameter_policy,
        )
        self.history.save_llm_exchange(iteration, "codex_edit_request", prompt, "")
        self._codex_checkpoint(candidate_workspace, "edit prompt saved to history", {"prompt_chars": len(prompt)})
        response = self._call_codex_exec(
            prompt,
            cwd=candidate_workspace,
            read_only=False,
            timeout=int(self.config.get("codex_cli", {}).get("edit_timeout_seconds", 3600)),
        )
        self.history.save_llm_exchange(iteration, "codex_edit", prompt, response)
        return response

    def _call_text_model(
        self,
        prompt: str,
        workspace: Path,
        read_only: bool,
        image_paths: list[Path] | None = None,
    ) -> str:
        if self.reasoning_backend == "openai_api":
            return self._call_openai_api(prompt, image_paths or [])
        if self.reasoning_backend == "codex_cli":
            return self._call_codex_exec(
                prompt,
                cwd=workspace,
                read_only=read_only,
                image_paths=image_paths or [],
            )
        raise ValueError(f"Unsupported reasoning_backend: {self.reasoning_backend}")

    def _call_openai_api(self, prompt: str, image_paths: list[Path] | None = None) -> str:
        api_key = resolve_openai_api_key()
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("Install openai or use llm_backend: codex_cli") from exc

        client = OpenAI(api_key=api_key)
        model = self.config.get("openai_model", "gpt-5.5")
        image_detail = str(self.config.get("openai_image_detail", "low"))
        try:
            if image_paths:
                content: list[dict[str, str]] = [{"type": "input_text", "text": prompt}]
                for image_path in image_paths:
                    image_bytes = image_path.read_bytes()
                    encoded = base64.b64encode(image_bytes).decode("ascii")
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{encoded}",
                            "detail": image_detail,
                        }
                    )
                response = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": content}],
                )
            else:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                )
            return response.output_text
        except AttributeError:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""

    def _call_codex_exec(
        self,
        prompt: str,
        cwd: Path,
        read_only: bool,
        timeout: int | None = None,
        image_paths: list[Path] | None = None,
    ) -> str:
        codex_cfg = self.config.get("codex_cli", {})
        command = self._resolve_codex_command(codex_cfg.get("command", "codex"))
        sandbox = "read-only" if read_only else codex_cfg.get("sandbox", "workspace-write")
        output_file = cwd / ".codex_last_message.txt"
        timeout_seconds = timeout or int(self.config.get("timeout_seconds", 3600))

        args = [
            command,
            "exec",
            "-C",
            str(cwd),
            "--sandbox",
            sandbox,
            "--output-last-message",
            str(output_file),
        ]
        if codex_cfg.get("skip_git_repo_check", True):
            args.append("--skip-git-repo-check")
        model = self.config.get("codex_model")
        if model:
            args.extend(["--model", model])
        for image_path in image_paths or []:
            args.extend(["--image", str(image_path)])
        args.extend(codex_cfg.get("extra_args", []))
        args.append("-")
        environment = os.environ.copy()
        environment.pop("OPENAI_API_KEY", None)
        environment.pop("API_KEY", None)
        stdout_path = cwd / ".codex_stdout.txt"
        stderr_path = cwd / ".codex_stderr.txt"
        command_log = {
            "command": command,
            "cwd": str(cwd),
            "read_only": read_only,
            "sandbox": sandbox,
            "timeout_seconds": timeout_seconds,
            "prompt_chars": len(prompt),
            "image_count": len(image_paths or []),
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "last_message_path": str(output_file),
            "args": args,
        }
        (cwd / ".codex_command.json").write_text(
            json.dumps(command_log, indent=2),
            encoding="utf-8",
            errors="replace",
        )
        self._codex_checkpoint(cwd, "command prepared", command_log)

        try:
            started_at = time.monotonic()
            self._codex_checkpoint(cwd, "opening stdout/stderr log files")
            with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
                "w",
                encoding="utf-8",
                errors="replace",
            ) as stderr_file:
                self._codex_checkpoint(cwd, "starting Codex CLI subprocess")
                process = subprocess.Popen(
                    args,
                    stdin=subprocess.PIPE,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    cwd=cwd,
                    env=environment,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                self._codex_checkpoint(cwd, "Codex CLI subprocess started", {"pid": process.pid})
                if process.stdin is None:
                    raise RuntimeError("Codex CLI stdin pipe was not opened.")
                self._codex_checkpoint(cwd, "writing prompt to Codex stdin")
                process.stdin.write(prompt)
                self._codex_checkpoint(cwd, "closing Codex stdin")
                process.stdin.close()
                self._codex_checkpoint(cwd, "waiting for Codex CLI to finish")

                last_progress = started_at
                while process.poll() is None:
                    now = time.monotonic()
                    if now - started_at > timeout_seconds:
                        self._codex_checkpoint(cwd, "timeout reached; killing Codex CLI", {"elapsed_seconds": int(now - started_at)})
                        process.kill()
                        raise subprocess.TimeoutExpired(args, timeout_seconds)
                    if now - last_progress >= self.progress_interval_seconds:
                        elapsed = int(now - started_at)
                        self._codex_checkpoint(cwd, "Codex CLI still running", {"elapsed_seconds": elapsed})
                        last_progress = now
                    time.sleep(1)
                return_code = process.returncode
                self._codex_checkpoint(cwd, "Codex CLI subprocess finished", {"return_code": return_code})
        except FileNotFoundError as exc:
            self._codex_checkpoint(cwd, "Codex CLI executable not found", {"command": command})
            raise RuntimeError(
                "Codex CLI executable was not found. On Linux, run `command -v codex` "
                "or locate the VS Code Server Codex binary, then set codex_cli.command "
                "in config.yaml to that full path. On Windows, run `where codex` and "
                "set codex_cli.command to the full codex.exe path."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            self._codex_checkpoint(cwd, "Codex CLI timed out", {"timeout_seconds": exc.timeout})
            (cwd / ".codex_timeout.txt").write_text(
                f"Codex CLI timed out after {exc.timeout} seconds.\n\n"
                f"STDOUT path: {stdout_path}\nSTDERR path: {stderr_path}",
                encoding="utf-8",
                errors="replace",
            )
            raise RuntimeError(f"Codex CLI timed out after {exc.timeout} seconds.") from exc
        if output_file.exists():
            self._codex_checkpoint(cwd, "reading Codex last-message output file")
            last_message = output_file.read_text(encoding="utf-8", errors="replace")
        else:
            self._codex_checkpoint(cwd, "last-message file missing; reading stdout fallback")
            last_message = stdout_path.read_text(encoding="utf-8", errors="replace")
        if return_code != 0:
            stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
            stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
            self._codex_checkpoint(
                cwd,
                "Codex CLI returned nonzero",
                {"return_code": return_code, "stdout_chars": len(stdout), "stderr_chars": len(stderr)},
            )
            raise RuntimeError(
                "Codex CLI failed with return code "
                f"{return_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            )
        self._codex_checkpoint(cwd, "Codex CLI edit response captured", {"response_chars": len(last_message)})
        return last_message

    def _codex_checkpoint(self, cwd: Path, stage: str, details: dict[str, Any] | None = None) -> None:
        details = details or {}
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] Codex edit checkpoint: {stage}"
        important_stage = (
            stage == "Codex CLI still running"
            or "timed out" in stage.lower()
            or "not found" in stage.lower()
            or "nonzero" in stage.lower()
        )
        if self.terminal_verbosity == "detailed" or important_stage:
            print(line, flush=True)
        progress_path = cwd / ".codex_progress.log"
        status_path = cwd / ".codex_status.json"
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        with progress_path.open("a", encoding="utf-8", errors="replace") as progress_file:
            progress_file.write(line + "\n")
            if details:
                progress_file.write(json.dumps(details, indent=2) + "\n")
        status_path.write_text(
            json.dumps(
                {
                    "timestamp": timestamp,
                    "stage": stage,
                    "details": details,
                },
                indent=2,
            ),
            encoding="utf-8",
            errors="replace",
        )

    def _resolve_codex_command(self, command: str) -> str:
        command_path = Path(command)
        if command_path.exists():
            return str(command_path)

        resolved = shutil.which(command)
        if resolved and Path(resolved).suffix.lower() != ".ps1":
            return resolved

        if command.lower() not in {"codex", "codex.exe"}:
            return command

        candidates = sorted(
            [
                *Path.home().glob(
                    ".vscode/extensions/openai.chatgpt-*/bin/windows-x86_64/codex.exe"
                ),
                *Path.home().glob(
                    ".vscode-server/extensions/openai.chatgpt-*/bin/linux-*/codex"
                ),
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return str(candidates[0])

        return command

    def _active_parameter_policy_text(self) -> str:
        value = self.config.get("calibration", {}).get("max_active_parameters")
        if value is None:
            return "No configured limit on the total number of active calibration parameters."
        return f"Keep the total number of active calibration parameters at or below {int(value)}."

    def _parse_candidate_json(self, text: str, expected_count: int = 2) -> dict[str, Any]:
        data = self._extract_json_object(text)
        candidates = data.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != expected_count:
            raise ValueError(f"LLM response must contain exactly {expected_count} candidates.")
        ids = {item.get("id") for item in candidates if isinstance(item, dict)}
        if data.get("recommended_candidate_id") not in ids:
            raise ValueError("recommended_candidate_id must match one candidate id.")
        return data

    def _parse_comparison_json(self, text: str) -> dict[str, Any]:
        data = self._extract_json_object(text)
        decision = str(data.get("decision", "")).lower().strip()
        if decision not in {"accepted", "rejected"}:
            raise ValueError("comparison decision must be accepted or rejected.")
        data["decision"] = decision
        if "reason" not in data:
            raise ValueError("comparison response must contain reason.")
        return data

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if not stripped.startswith("{"):
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start >= 0 and end > start:
                stripped = stripped[start : end + 1]
        data = json.loads(stripped)
        return data

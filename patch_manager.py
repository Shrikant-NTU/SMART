from __future__ import annotations

import ast
import difflib
import fnmatch
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from parameter_patch_manager import ParameterPatchManager
from parameter_space import (
    CALIBRATION_MANIFEST_NAME,
    ensure_calibration_manifest,
    load_parameter_specs,
    update_manifest_defaults,
)

DEFAULT_IMMUTABLE_RESULT_MARKERS = (
    "immutable",
    "read-only",
    "readonly",
    "read_only",
    "frozen",
    "tuple(",
    "frozenset(",
    "mappingproxytype(",
    "jnp.",
    "jax.",
    "tf.",
    "tensorflow.",
)

EDITABLE_CANDIDATE_FILES = {
    "jsfm/sfm.py",
    CALIBRATION_MANIFEST_NAME,
}

CODEX_RUNTIME_FILE_NAMES = {
    ".codex_command.json",
    ".codex_last_message.txt",
    ".codex_progress.log",
    ".codex_status.json",
    ".codex_stderr.txt",
    ".codex_stdout.txt",
    ".codex_timeout.txt",
}


def _top_level_symbols(tree: ast.Module) -> dict[str, ast.AST]:
    symbols: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols[node.name] = node
            continue
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols[target.id] = node
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            symbols[node.target.id] = node
    return symbols


def _reachable_symbols(symbols: dict[str, ast.AST], root: str) -> set[str]:
    reachable: set[str] = set()
    pending = [root]
    while pending:
        name = pending.pop()
        if name in reachable or name not in symbols:
            continue
        reachable.add(name)
        dependencies = {
            node.id
            for node in ast.walk(symbols[name])
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in symbols
        }
        pending.extend(sorted(dependencies - reachable))
    return reachable


def _ast_fingerprint(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def _direct_nested_function(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in function.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _named_assignment_fingerprints(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    name: str,
) -> list[str]:
    fingerprints: list[str] = []
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            fingerprints.append(_ast_fingerprint(node) or "")
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            fingerprints.append(_ast_fingerprint(node) or "")
        elif (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            fingerprints.append(_ast_fingerprint(node) or "")
    return fingerprints


def _return_fingerprints(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    recursive: bool,
) -> list[str]:
    nodes = ast.walk(function) if recursive else function.body
    return [
        _ast_fingerprint(node) or ""
        for node in nodes
        if isinstance(node, ast.Return)
    ]


def _other_top_level_fingerprints(tree: ast.Module) -> list[str]:
    fingerprints: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Assign) and all(
            isinstance(target, ast.Name) for target in node.targets
        ):
            continue
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            continue
        fingerprints.append(ast.dump(node, annotate_fields=True, include_attributes=False))
    return fingerprints


class PatchManager:
    """Creates isolated workspaces, validates code, and archives versions."""

    def __init__(self, config: dict[str, Any], project_dir: Path):
        self.config = config
        self.project_dir = project_dir
        self.source_project_dir = self._resolve_path(config["source_project_dir"]).resolve()
        self.working_root = project_dir / "working_model"
        self.accepted_root = project_dir / "accepted_versions"
        self.rejected_root = project_dir / "rejected_versions"
        self.python_executable = self._resolve_path(config.get("python_executable", "python"))
        self.exclude_names = set(
            config.get(
                "copy_exclude_names",
                [
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".images",
                    "calibration",
                    "Old experiments",
                ],
            )
        )
        self.exclude_patterns = list(config.get("copy_exclude_patterns", ["*.zip", "*.pdf"]))

    def prepare_initial_workspace(self, examples: list[int] | None = None) -> Path:
        workspace = self.working_root / self._unique_name("baseline")
        self.copy_candidate_files(self.source_project_dir, workspace, examples)
        self.ensure_calibration_manifest(workspace)
        return workspace

    def prepare_resume_workspace(self, examples: list[int] | None = None) -> tuple[Path, Path]:
        source = self.calibration_source(examples)
        workspace = self.working_root / self._unique_name("resume_seed")
        self.copy_candidate_files(source, workspace, examples)
        self.ensure_calibration_manifest(workspace)
        return workspace, source

    def prepare_calibration_workspace(self, examples: list[int] | None = None) -> Path:
        workspace = self.working_root / self._unique_name("calibration_seed")
        self.copy_candidate_files(self.calibration_source(examples), workspace, examples)
        self.ensure_calibration_manifest(workspace)
        return workspace

    def create_candidate_workspace(
        self,
        current_workspace: Path,
        iteration: int,
        examples: list[int] | None = None,
    ) -> Path:
        workspace = self.working_root / self._unique_name(f"iteration_{iteration:03d}_candidate")
        self.copy_candidate_files(current_workspace, workspace, examples)
        self.ensure_calibration_manifest(workspace)
        return workspace

    def copy_candidate_files(
        self,
        source: Path,
        destination: Path,
        examples: list[int] | None = None,
    ) -> None:
        """Copy only files needed to edit, validate, and run an iteration candidate."""
        destination.mkdir(parents=True, exist_ok=False)

        relative_paths: set[Path] = {
            Path(self.config.get("model_file_name", "jsfm/sfm.py")),
            Path(CALIBRATION_MANIFEST_NAME),
        }
        runner_script = self.config.get("runner_script_name")
        if runner_script:
            relative_paths.add(Path(runner_script))

        if examples:
            relative_paths.update(Path(f"example{number}.py") for number in examples)
        else:
            example_glob = self.config.get("example_glob", "example*.py")
            relative_paths.update(path.relative_to(source) for path in source.glob(example_glob))

        for configured_path in self.config.get("candidate_workspace_files", []):
            relative_paths.add(Path(configured_path))
        for pattern in self.config.get("additional_tracked_globs", []):
            relative_paths.update(path.relative_to(source) for path in source.glob(pattern))

        directory_paths = {
            Path(value)
            for value in self.config.get("candidate_workspace_directories", ["jsfm"])
        }
        for relative in sorted(directory_paths):
            source_path = source / relative
            if not source_path.is_dir() or self._excluded(relative):
                continue
            shutil.copytree(
                source_path,
                destination / relative,
                ignore=self._copy_ignore,
            )

        for relative in sorted(relative_paths):
            source_path = source / relative
            if not source_path.is_file() or self._excluded(relative):
                continue
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target)

    def copy_project(self, source: Path, destination: Path) -> None:
        shutil.copytree(source, destination, ignore=self._copy_ignore)

    def _copy_ignore(self, _directory: str, names: list[str]) -> set[str]:
        ignored = {name for name in names if name in self.exclude_names}
        for pattern in self.exclude_patterns:
            ignored.update(name for name in names if fnmatch.fnmatch(name, pattern))
        return ignored

    def _excluded(self, relative: Path) -> bool:
        return any(part in self.exclude_names for part in relative.parts) or any(
            fnmatch.fnmatch(relative.name, pattern) for pattern in self.exclude_patterns
        )

    def tracked_files(self, workspace: Path) -> list[Path]:
        files = [
            workspace / self.config.get("model_file_name", "jsfm/sfm.py"),
            workspace / CALIBRATION_MANIFEST_NAME,
        ]
        files.extend(sorted(workspace.glob(self.config.get("example_glob", "example*.py"))))
        for pattern in self.config.get("additional_tracked_globs", []):
            files.extend(sorted(workspace.glob(pattern)))

        seen: set[Path] = set()
        tracked: list[Path] = []
        for path in files:
            if path.exists() and path not in seen:
                seen.add(path)
                tracked.append(path)
        return tracked

    def capture_edit_scope(self, workspace: Path) -> dict[str, str]:
        """Record candidate files immediately before Codex is allowed to edit."""
        snapshot: dict[str, str] = {}
        for path in sorted(workspace.rglob("*")):
            if not path.is_file() or self._ignore_edit_scope_path(path, workspace):
                continue
            relative = path.relative_to(workspace).as_posix()
            snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return snapshot

    def validate_edit_scope(
        self,
        workspace: Path,
        baseline: dict[str, str],
    ) -> dict[str, Any]:
        """Allow modifications only to the model file and calibration manifest."""
        current = self.capture_edit_scope(workspace)
        baseline_paths = set(baseline)
        current_paths = set(current)
        created = sorted(current_paths - baseline_paths)
        deleted = sorted(baseline_paths - current_paths)
        modified = sorted(
            path
            for path in baseline_paths & current_paths
            if baseline[path] != current[path]
        )
        allowed_modified = sorted(
            path for path in modified if path in EDITABLE_CANDIDATE_FILES
        )
        disallowed_modified = sorted(
            path for path in modified if path not in EDITABLE_CANDIDATE_FILES
        )

        violations = [
            *[f"modified:{path}" for path in disallowed_modified],
            *[f"created:{path}" for path in created],
            *[f"deleted:{path}" for path in deleted],
        ]
        result: dict[str, Any] = {
            "ok": not violations,
            "editable_files": sorted(EDITABLE_CANDIDATE_FILES),
            "allowed_modified_files": allowed_modified,
            "disallowed_modified_files": disallowed_modified,
            "created_files": created,
            "deleted_files": deleted,
        }
        if violations:
            result["error"] = (
                "Codex edit scope violation; only existing allowlisted files may be modified: "
                + ", ".join(violations)
            )
        return result

    def validate_sfm_function_scope(
        self,
        parent_workspace: Path,
        child_workspace: Path,
    ) -> dict[str, Any]:
        """Restrict sfm.py edits to the single_update_active dependency closure."""
        relative = Path(self.config.get("model_file_name", "jsfm/sfm.py"))
        parent_path = parent_workspace / relative
        child_path = child_workspace / relative
        try:
            parent_tree = ast.parse(parent_path.read_text(encoding="utf-8"))
            child_tree = ast.parse(child_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            return {
                "ok": False,
                "root_function": "single_update_active",
                "error": f"Unable to validate {relative.as_posix()} function scope: {exc}",
            }

        parent_symbols = _top_level_symbols(parent_tree)
        child_symbols = _top_level_symbols(child_tree)
        if "single_update_active" not in parent_symbols:
            return {
                "ok": False,
                "root_function": "single_update_active",
                "error": f"Parent {relative.as_posix()} has no single_update_active function.",
            }
        if "single_update_active" not in child_symbols:
            return {
                "ok": False,
                "root_function": "single_update_active",
                "error": f"Child {relative.as_posix()} removed single_update_active.",
            }

        parent_reachable = _reachable_symbols(parent_symbols, "single_update_active")
        child_reachable = _reachable_symbols(child_symbols, "single_update_active")
        changed_symbols = sorted(
            name
            for name in set(parent_symbols) | set(child_symbols)
            if _ast_fingerprint(parent_symbols.get(name))
            != _ast_fingerprint(child_symbols.get(name))
        )
        allowed_symbols = parent_reachable | child_reachable
        allowed_changed = sorted(name for name in changed_symbols if name in allowed_symbols)
        disallowed_changed = sorted(name for name in changed_symbols if name not in allowed_symbols)

        module_structure_changed = (
            _other_top_level_fingerprints(parent_tree)
            != _other_top_level_fingerprints(child_tree)
        )
        prescribed_update = self._validate_prescribed_state_update(
            relative,
            child_tree,
        )
        result: dict[str, Any] = {
            "ok": (
                not disallowed_changed
                and not module_structure_changed
                and prescribed_update["ok"]
            ),
            "root_function": "single_update_active",
            "parent_reachable_symbols": sorted(parent_reachable),
            "child_reachable_symbols": sorted(child_reachable),
            "allowed_changed_symbols": allowed_changed,
            "disallowed_changed_symbols": disallowed_changed,
            "protected_module_structure_changed": module_structure_changed,
            "prescribed_state_update_lock": prescribed_update,
        }
        violations: list[str] = []
        if disallowed_changed:
            violations.append("unrelated symbols changed: " + ", ".join(disallowed_changed))
        if module_structure_changed:
            violations.append("imports or unsupported top-level statements changed")
        if not prescribed_update["ok"]:
            violations.append(prescribed_update["error"])
        if violations:
            result["error"] = (
                "SFM function-scope violation; edits must remain in the "
                "single_update_active dependency pathway: " + "; ".join(violations)
            )
        return result

    def _validate_prescribed_state_update(
        self,
        relative: Path,
        child_tree: ast.Module,
    ) -> dict[str, Any]:
        policy = self.config.get("prescribed_state_update_lock", {})
        if not policy.get("enabled", False):
            return {"ok": True, "enabled": False, "violations": []}

        baseline_path = self.source_project_dir / relative
        try:
            baseline_tree = ast.parse(baseline_path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as exc:
            return {
                "ok": False,
                "enabled": True,
                "violations": [f"scientist baseline unavailable: {exc}"],
                "error": f"Scientist-prescribed state-update check failed: {exc}",
            }

        baseline_symbols = _top_level_symbols(baseline_tree)
        child_symbols = _top_level_symbols(child_tree)
        violations: list[str] = []

        protected_functions = [
            str(name) for name in policy.get("protected_functions", [])
        ]
        for name in protected_functions:
            baseline_node = baseline_symbols.get(name)
            child_node = child_symbols.get(name)
            if baseline_node is None:
                violations.append(f"baseline protected function missing: {name}")
            elif child_node is None:
                violations.append(f"protected function removed: {name}")
            elif _ast_fingerprint(baseline_node) != _ast_fingerprint(child_node):
                violations.append(f"protected function changed: {name}")

        update_name = str(policy.get("update_function", "single_update_active"))
        active_body_name = str(
            policy.get("active_body_function", "update_active_human")
        )
        baseline_update = baseline_symbols.get(update_name)
        child_update = child_symbols.get(update_name)
        if not isinstance(baseline_update, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.append(f"baseline update function missing: {update_name}")
            baseline_active_body = None
        else:
            baseline_active_body = _direct_nested_function(
                baseline_update,
                active_body_name,
            )
        if not isinstance(child_update, (ast.FunctionDef, ast.AsyncFunctionDef)):
            violations.append(f"child update function missing: {update_name}")
            child_active_body = None
        else:
            child_active_body = _direct_nested_function(child_update, active_body_name)

        if baseline_active_body is None:
            violations.append(
                f"baseline active-update body missing: {active_body_name}"
            )
        if child_active_body is None:
            violations.append(f"active-update body removed: {active_body_name}")

        if baseline_active_body is not None and child_active_body is not None:
            for name in policy.get("protected_assignments", []):
                assignment_name = str(name)
                baseline_assignments = _named_assignment_fingerprints(
                    baseline_active_body,
                    assignment_name,
                )
                child_assignments = _named_assignment_fingerprints(
                    child_active_body,
                    assignment_name,
                )
                if len(baseline_assignments) != 1:
                    violations.append(
                        "baseline must contain exactly one protected assignment: "
                        + assignment_name
                    )
                elif child_assignments != baseline_assignments:
                    violations.append(
                        f"protected assignment changed or duplicated: {assignment_name}"
                    )

            if policy.get("protect_active_body_returns", True):
                if _return_fingerprints(
                    child_active_body,
                    recursive=True,
                ) != _return_fingerprints(
                    baseline_active_body,
                    recursive=True,
                ):
                    violations.append(
                        f"protected return changed in {active_body_name}"
                    )

        if (
            policy.get("protect_update_function_returns", True)
            and isinstance(baseline_update, (ast.FunctionDef, ast.AsyncFunctionDef))
            and isinstance(child_update, (ast.FunctionDef, ast.AsyncFunctionDef))
            and _return_fingerprints(child_update, recursive=False)
            != _return_fingerprints(baseline_update, recursive=False)
        ):
            violations.append(f"protected return changed in {update_name}")

        result: dict[str, Any] = {
            "ok": not violations,
            "enabled": True,
            "baseline": str(baseline_path),
            "protected_functions": protected_functions,
            "protected_assignments": [
                str(name) for name in policy.get("protected_assignments", [])
            ],
            "violations": violations,
        }
        if violations:
            result["error"] = (
                "Scientist-prescribed state-update equations changed: "
                + ", ".join(violations)
            )
        return result

    def _ignore_edit_scope_path(self, path: Path, workspace: Path) -> bool:
        relative = path.relative_to(workspace)
        if path.name in CODEX_RUNTIME_FILE_NAMES:
            return True
        return self._excluded(relative)

    def validate(self, workspace: Path) -> dict[str, Any]:
        files = self.tracked_files(workspace)
        python_files = [path for path in files if path.suffix == ".py"]
        compile_cmd = [str(self.python_executable), "-m", "py_compile", *[str(path) for path in python_files]]
        compile_result = subprocess.run(
            compile_cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        import_results = []
        for module_name in self.config.get("validation_imports", ["jsfm.sfm"]):
            import_results.append(
                subprocess.run(
                    [str(self.python_executable), "-c", f"import {module_name}; print('import ok: {module_name}')"],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            )
        imports_ok = all(result.returncode == 0 for result in import_results)
        manifest_error = None
        try:
            specs = load_parameter_specs(self.config, workspace)
            ParameterPatchManager().validate_targets(workspace, specs, require_default_match=True)
        except (OSError, ValueError) as exc:
            manifest_error = str(exc)
        contract_errors = self._interface_contract_errors(workspace)
        return {
            "ok": (
                compile_result.returncode == 0
                and imports_ok
                and manifest_error is None
                and not contract_errors
            ),
            "compile_returncode": compile_result.returncode,
            "compile_stdout": compile_result.stdout,
            "compile_stderr": compile_result.stderr,
            "import_results": [
                {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
                for result in import_results
            ],
            "calibration_manifest": str(workspace / CALIBRATION_MANIFEST_NAME),
            "calibration_manifest_error": manifest_error,
            "contract_errors": contract_errors,
            "array_boundary_errors": contract_errors,
        }

    def _interface_contract_errors(self, workspace: Path) -> list[str]:
        immutable_result_functions = self._immutable_result_function_names(workspace)
        if not immutable_result_functions:
            return []

        errors: list[str] = []
        module_prefixes = self._contract_module_prefixes()
        for path in sorted(workspace.glob(self.config.get("example_glob", "example*.py"))):
            try:
                source = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source, filename=str(path))
            except SyntaxError:
                continue

            callable_names = self._import_aliases_for(tree, immutable_result_functions, module_prefixes)
            immutable_assigned_variables: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign) and self._is_call_to(node.value, callable_names):
                    for target in node.targets:
                        immutable_assigned_variables.update(self._target_names(target))
                elif isinstance(node, ast.AnnAssign) and self._is_call_to(node.value, callable_names):
                    immutable_assigned_variables.update(self._target_names(node.target))

            if not immutable_assigned_variables:
                continue

            for node in ast.walk(tree):
                target = None
                if isinstance(node, ast.Assign):
                    for assign_target in node.targets:
                        target = self._subscript_base_name(assign_target)
                        if target in immutable_assigned_variables:
                            errors.append(self._interface_contract_message(workspace, path, node.lineno, target))
                elif isinstance(node, ast.AugAssign):
                    target = self._subscript_base_name(node.target)
                    if target in immutable_assigned_variables:
                        errors.append(self._interface_contract_message(workspace, path, node.lineno, target))

        return errors

    def _immutable_result_function_names(self, workspace: Path) -> set[str]:
        markers = self._immutable_result_markers()
        function_names: set[str] = set()
        for source_root in self._contract_source_roots(workspace):
            if source_root.is_file() and source_root.suffix == ".py":
                source_files = [source_root]
            elif source_root.is_dir():
                source_files = sorted(source_root.rglob("*.py"))
            else:
                source_files = []

            for model_file in source_files:
                source = model_file.read_text(encoding="utf-8", errors="replace")
                try:
                    tree = ast.parse(source, filename=str(model_file))
                except SyntaxError:
                    continue

                for node in ast.walk(tree):
                    if not isinstance(node, ast.FunctionDef):
                        continue
                    function_source = ast.get_source_segment(source, node) or ""
                    if self._function_matches_immutable_result_marker(node, function_source, markers):
                        function_names.add(node.name)
        return function_names

    def _contract_source_roots(self, workspace: Path) -> list[Path]:
        contract_config = self.config.get("contract_validation", {})
        configured_roots = contract_config.get("source_roots")
        if configured_roots is None:
            configured_roots = self.config.get("candidate_workspace_directories", ["jsfm"])
        return [workspace / Path(value) for value in configured_roots]

    def _contract_module_prefixes(self) -> tuple[str, ...]:
        contract_config = self.config.get("contract_validation", {})
        configured_prefixes = contract_config.get("module_prefixes")
        if configured_prefixes is None:
            configured_roots = contract_config.get("source_roots")
            if configured_roots is None:
                configured_roots = self.config.get("candidate_workspace_directories", ["jsfm"])
            configured_prefixes = [
                Path(value).parts[0]
                for value in configured_roots
                if Path(value).parts
            ]
        return tuple(str(prefix) for prefix in configured_prefixes)

    def _immutable_result_markers(self) -> tuple[str, ...]:
        contract_config = self.config.get("contract_validation", {})
        configured_markers = contract_config.get("immutable_result_markers")
        markers = configured_markers or DEFAULT_IMMUTABLE_RESULT_MARKERS
        return tuple(str(marker).lower() for marker in markers)

    @staticmethod
    def _function_matches_immutable_result_marker(
        node: ast.FunctionDef,
        source: str,
        markers: tuple[str, ...],
    ) -> bool:
        source_lower = source.lower()
        if any(marker in source_lower for marker in markers):
            return True
        for child in ast.walk(node):
            name = PatchManager._call_name(child).lower()
            if name and any(marker.strip(".(") == name for marker in markers):
                return True
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if any(marker in child.value.lower() for marker in markers):
                    return True
        return False

    @staticmethod
    def _import_aliases_for(
        tree: ast.AST,
        function_names: set[str],
        module_prefixes: tuple[str, ...],
    ) -> set[str]:
        names = set(function_names)
        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.ImportFrom)
                or node.module is None
                or not any(node.module == prefix or node.module.startswith(f"{prefix}.") for prefix in module_prefixes)
            ):
                continue
            for alias in node.names:
                if alias.name in function_names:
                    names.add(alias.asname or alias.name)
        return names

    @staticmethod
    def _is_call_to(node: ast.AST | None, callable_names: set[str]) -> bool:
        if not isinstance(node, ast.Call):
            return False
        name = PatchManager._call_name(node.func)
        return name in callable_names or name.split(".")[-1] in callable_names

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = PatchManager._call_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""

    @staticmethod
    def _target_names(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Name):
            return {node.id}
        if isinstance(node, (ast.Tuple, ast.List)):
            names: set[str] = set()
            for element in node.elts:
                names.update(PatchManager._target_names(element))
            return names
        return set()

    @staticmethod
    def _subscript_base_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            return node.value.id
        return None

    @staticmethod
    def _interface_contract_message(workspace: Path, path: Path, line: int, name: str) -> str:
        relative = path.relative_to(workspace)
        return (
            f"{relative}:{line}: {name} is assigned from a helper function that "
            "appears to return an immutable or backend-owned result, then later "
            "mutated with item assignment. Preserve the caller's expected data "
            "contract or update the whole downstream workflow consistently."
        )

    def ensure_calibration_manifest(self, workspace: Path) -> Path:
        path = workspace / CALIBRATION_MANIFEST_NAME
        created = not path.exists()
        path = ensure_calibration_manifest(workspace, self.config)
        if created:
            specs = load_parameter_specs(self.config, workspace)
            source_values = ParameterPatchManager().read_values(workspace, specs)
            update_manifest_defaults(workspace, self.config, source_values)
        return path

    def archive_version(
        self,
        workspace: Path,
        accepted: bool,
        iteration: int,
        metadata: dict[str, Any],
    ) -> Path:
        root = self.accepted_root if accepted else self.rejected_root
        destination = root / self._unique_name(f"iteration_{iteration:03d}")
        destination.mkdir(parents=True, exist_ok=False)
        source_snapshot = destination / "source"
        source_snapshot.mkdir(parents=True, exist_ok=True)

        for path in self.tracked_files(workspace):
            relative = path.relative_to(workspace)
            target = source_snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)

        if self.config.get("history", {}).get("keep_full_workspace_snapshots", False):
            self.copy_project(workspace, destination / "full_workspace")

        (destination / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        return destination

    def diff(self, before: Path, after: Path) -> str:
        chunks: list[str] = []
        before_files = {path.relative_to(before): path for path in self.tracked_files(before)}
        after_files = {path.relative_to(after): path for path in self.tracked_files(after)}
        for relative in sorted(set(before_files) | set(after_files)):
            old_text = (
                before_files[relative].read_text(encoding="utf-8", errors="replace").splitlines(True)
                if relative in before_files
                else []
            )
            new_text = (
                after_files[relative].read_text(encoding="utf-8", errors="replace").splitlines(True)
                if relative in after_files
                else []
            )
            if old_text == new_text:
                continue
            chunks.extend(
                difflib.unified_diff(
                    old_text,
                    new_text,
                    fromfile=f"a/{relative.as_posix()}",
                    tofile=f"b/{relative.as_posix()}",
                )
            )
        return "".join(chunks)

    def export_seed_files(
        self,
        source: Path | None = None,
        examples: list[int] | None = None,
    ) -> None:
        """Keep the runnable subset of the current seed for human inspection."""
        self.working_root.mkdir(parents=True, exist_ok=True)
        seed_workspace = self.working_root / "current_seed"
        if seed_workspace.exists():
            shutil.rmtree(seed_workspace)
        seed_source = source or self.source_project_dir
        self.copy_candidate_files(seed_source, seed_workspace, examples)
        self.ensure_calibration_manifest(seed_workspace)
        (self.working_root / "README.md").write_text(
            "current_seed/ is a focused runnable copy of the source used to seed "
            "the latest controller run. Runtime baseline_*, resume_seed_* and "
            "iteration_* folders are separate focused execution workspaces.\n",
            encoding="utf-8",
        )

    def latest_accepted_source(self) -> Path | None:
        if not self.accepted_root.exists():
            return None
        candidates = [
            path / "source"
            for path in self.accepted_root.iterdir()
            if path.is_dir() and (path / "source").exists()
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.parent.stat().st_mtime)

    def calibration_source(self, examples: list[int] | None = None) -> Path:
        latest = self.latest_accepted_source()
        if latest and self._source_has_examples(latest, examples):
            return latest
        return self.source_project_dir

    @staticmethod
    def _source_has_examples(source: Path, examples: list[int] | None) -> bool:
        if not examples:
            return True
        return all((source / f"example{number}.py").exists() for number in examples)

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
    def _unique_name(prefix: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{prefix}_{timestamp}"

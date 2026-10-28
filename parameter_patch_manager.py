from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from typing import Any

from parameter_space import ParameterSpec


class ParameterPatchManager:
    """Reads and patches validated scalar calibration targets in a workspace."""

    DEFAULT_MATCH_REL_TOL = 1e-6
    DEFAULT_MATCH_ABS_TOL = 1e-6

    def apply(
        self,
        workspace: Path,
        specs: list[ParameterSpec],
        values: dict[str, float],
        fixed_values: dict[str, float] | None = None,
    ) -> None:
        del fixed_values  # Retained in the public signature for older callers.
        grouped: dict[str, list[ParameterSpec]] = {}
        for spec in specs:
            if spec.name in values:
                grouped.setdefault(spec.file, []).append(spec)

        for relative_file, file_specs in grouped.items():
            path = workspace / relative_file
            if not path.is_file():
                raise FileNotFoundError(f"Calibration target file not found: {path}")
            text = path.read_text(encoding="utf-8")
            indexed = [spec for spec in file_specs if spec.target_type == "single_params_index"]
            constants = [spec for spec in file_specs if spec.target_type == "module_constant"]
            if indexed:
                text = self._patch_single_params(text, path, indexed, values)
            for spec in constants:
                text = self._patch_module_constant(text, path, spec, spec.clip(values[spec.name]))
            path.write_text(text, encoding="utf-8")

    def validate_targets(
        self,
        workspace: Path,
        specs: list[ParameterSpec],
        require_default_match: bool = True,
    ) -> dict[str, float | int]:
        current = self.read_values(workspace, specs)
        if require_default_match:
            for spec in specs:
                actual = float(current[spec.name])
                if not math.isclose(
                    actual,
                    float(spec.default),
                    rel_tol=self.DEFAULT_MATCH_REL_TOL,
                    abs_tol=self.DEFAULT_MATCH_ABS_TOL,
                ):
                    raise ValueError(
                        f"Calibration parameter {spec.name} declares default {spec.default}, "
                        f"but its source target currently equals {current[spec.name]}."
                    )
        return current

    def read_values(self, workspace: Path, specs: list[ParameterSpec]) -> dict[str, float | int]:
        values: dict[str, float | int] = {}
        text_by_file: dict[str, str] = {}
        for spec in specs:
            path = workspace / spec.file
            if not path.is_file():
                raise FileNotFoundError(f"Calibration target file not found: {path}")
            text = text_by_file.setdefault(spec.file, path.read_text(encoding="utf-8"))
            if spec.target_type == "single_params_index":
                value = self._read_single_params_value(text, path, int(spec.index))
            else:
                value = self._read_module_constant(text, path, str(spec.target_name))
            values[spec.name] = int(round(value)) if spec.data_type == "int" else value
        return values

    @staticmethod
    def _single_params_match(text: str, path: Path) -> re.Match[str]:
        match = re.search(
            r"single_params\s*=\s*jnp\.array\(\s*\[(?P<body>.*?)\]\s*\)",
            text,
            flags=re.DOTALL,
        )
        if not match:
            raise ValueError(f"Could not find single_params array in {path}")
        return match

    def _patch_single_params(
        self,
        text: str,
        path: Path,
        specs: list[ParameterSpec],
        values: dict[str, float],
    ) -> str:
        match = self._single_params_match(text, path)
        lines = match.group("body").splitlines()
        for spec in specs:
            lines = self._replace_indexed_value(lines, int(spec.index), spec.clip(values[spec.name]))
        body = "\n".join(lines)
        return text[: match.start("body")] + body + text[match.end("body") :]

    def _read_single_params_value(self, text: str, path: Path, target_index: int) -> float:
        match = self._single_params_match(text, path)
        values = self._indexed_value_tokens(match.group("body").splitlines())
        if target_index >= len(values):
            raise ValueError(f"single_params array in {path} has no index {target_index}")
        try:
            return float(values[target_index])
        except ValueError as exc:
            raise ValueError(
                f"single_params index {target_index} in {path} must be a numeric literal"
            ) from exc

    @staticmethod
    def _indexed_value_tokens(lines: list[str]) -> list[str]:
        values: list[str] = []
        pattern = re.compile(r"^\s*(?P<value>[^,#]+?)\s*,")
        for line in lines:
            match = pattern.match(line)
            if match:
                values.append(match.group("value").strip())
        return values

    @staticmethod
    def _replace_indexed_value(
        lines: list[str],
        target_index: int,
        value: float | int,
    ) -> list[str]:
        current_index = -1
        output: list[str] = []
        pattern = re.compile(r"(?P<prefix>^\s*)(?P<value>[^,#]+?)(?P<suffix>\s*,.*$)")
        replacement = str(value) if isinstance(value, int) else f"{float(value):.9g}"
        for line in lines:
            match = pattern.match(line)
            if match:
                current_index += 1
                if current_index == target_index:
                    output.append(f"{match.group('prefix')}{replacement}{match.group('suffix')}")
                    continue
            output.append(line)
        if current_index < target_index:
            raise ValueError(f"single_params array has no index {target_index}")
        return output

    def _patch_module_constant(
        self,
        text: str,
        path: Path,
        spec: ParameterSpec,
        value: float | int,
    ) -> str:
        node = self._module_constant_value_node(text, path, str(spec.target_name))
        lines = text.splitlines(keepends=True)
        if node.lineno != node.end_lineno:
            raise ValueError(f"Module constant {spec.target_name} in {path} must be on one line")
        line_index = node.lineno - 1
        replacement = str(value) if isinstance(value, int) else f"{float(value):.9g}"
        line = lines[line_index]
        lines[line_index] = line[: node.col_offset] + replacement + line[node.end_col_offset :]
        return "".join(lines)

    def _read_module_constant(self, text: str, path: Path, name: str) -> float:
        node = self._module_constant_value_node(text, path, name)
        value = ast.literal_eval(node)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Module constant {name} in {path} is not a numeric literal")
        return float(value)

    @staticmethod
    def _module_constant_value_node(text: str, path: Path, name: str) -> ast.expr:
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as exc:
            raise ValueError(f"Could not parse calibration target file {path}: {exc}") from exc
        matches: list[ast.expr] = []
        for statement in tree.body:
            if isinstance(statement, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == name for target in statement.targets
            ):
                matches.append(statement.value)
            elif (
                isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id == name
                and statement.value is not None
            ):
                matches.append(statement.value)
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one top-level assignment for {name} in {path}")
        try:
            value = ast.literal_eval(matches[0])
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Module constant {name} in {path} must be a numeric literal") from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Module constant {name} in {path} must be a numeric literal")
        return matches[0]

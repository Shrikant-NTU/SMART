from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml


CALIBRATION_MANIFEST_NAME = "calibration_parameters.yaml"
SUPPORTED_TARGET_TYPES = {"single_params_index", "module_constant"}
SUPPORTED_DATA_TYPES = {"float", "int"}
PARAMETER_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    default: float
    minimum: float
    maximum: float
    mutation_size: float
    target_type: str
    file: str
    index: int | None = None
    target_name: str | None = None
    data_type: str = "float"

    def clip(self, value: float) -> float | int:
        clipped = min(self.maximum, max(self.minimum, float(value)))
        if self.data_type == "int":
            return int(round(clipped))
        return clipped

    def ga_spec(self) -> dict[str, Any]:
        return {
            "default": self.default,
            "lower": self.minimum,
            "upper": self.maximum,
            "index": self.index,
            "data_type": self.data_type,
            "target_type": self.target_type,
            "file": self.file,
            "target_name": self.target_name,
        }


DEFAULT_PARAMETER_SPECS = [
    ParameterSpec("tau", 0.5, 0.2, 1.2, 0.15, "single_params_index", "jsfm/utils.py", index=3),
    ParameterSpec("Ai", 2.1, 1.0, 4.2, 0.24, "single_params_index", "jsfm/utils.py", index=4),
    ParameterSpec("Bi", 0.3, 0.1, 0.6, 0.075, "single_params_index", "jsfm/utils.py", index=6),
]


def configured_default_specs(config: dict[str, Any]) -> list[ParameterSpec]:
    calibration = config.get("calibration", {})
    ranges = calibration.get("parameter_ranges", {})
    mutation_sizes = calibration.get("parameter_mutation_sizes", {})
    specs: list[ParameterSpec] = []
    for base in DEFAULT_PARAMETER_SPECS:
        override = ranges.get(base.name, {})
        specs.append(
            replace(
                base,
                default=float(override.get("default", base.default)),
                minimum=float(override.get("min", base.minimum)),
                maximum=float(override.get("max", base.maximum)),
                mutation_size=float(mutation_sizes.get(base.name, base.mutation_size)),
                file=str(calibration.get("parameter_file", base.file)),
            )
        )
    _validate_specs(specs, config)
    return specs


def load_parameter_specs(
    config: dict[str, Any],
    workspace: Path | None = None,
) -> list[ParameterSpec]:
    manifest_path = workspace / CALIBRATION_MANIFEST_NAME if workspace else None
    if manifest_path is None or not manifest_path.is_file():
        return configured_default_specs(config)

    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in calibration manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Calibration manifest must contain a YAML mapping: {manifest_path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"Calibration manifest schema_version must be 1: {manifest_path}")
    entries = payload.get("parameters")
    if not isinstance(entries, list):
        raise ValueError(f"Calibration manifest parameters must be a list: {manifest_path}")

    if any(not isinstance(entry, dict) for entry in entries):
        raise ValueError(f"Every calibration manifest parameter must be a mapping: {manifest_path}")
    specs = [
        _spec_from_manifest_entry(entry, manifest_path)
        for entry in entries
        if entry.get("calibrate", True)
    ]
    if not specs:
        raise ValueError(f"Calibration manifest has no active numeric parameters: {manifest_path}")
    _validate_specs(specs, config)
    return specs


def ensure_calibration_manifest(workspace: Path, config: dict[str, Any]) -> Path:
    path = workspace / CALIBRATION_MANIFEST_NAME
    if not path.exists():
        write_parameter_manifest(path, configured_default_specs(config))
    else:
        load_parameter_specs(config, workspace)
    return path


def write_parameter_manifest(path: Path, specs: list[ParameterSpec]) -> None:
    payload = {
        "schema_version": 1,
        "parameters": [_manifest_entry(spec) for spec in specs],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def update_manifest_defaults(
    workspace: Path,
    config: dict[str, Any],
    values: dict[str, float],
) -> Path:
    path = ensure_calibration_manifest(workspace, config)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    for entry in payload.get("parameters", []):
        if not isinstance(entry, dict) or entry.get("name") not in values:
            continue
        data_type = entry.get("data_type", "float")
        value = float(values[entry["name"]])
        entry["default"] = int(round(value)) if data_type == "int" else value
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    load_parameter_specs(config, workspace)
    return path


def default_parameter_values(specs: list[ParameterSpec]) -> dict[str, float | int]:
    return {spec.name: spec.clip(spec.default) for spec in specs}


def validate_manifest_inheritance(
    parent_workspace: Path,
    child_workspace: Path,
    config: dict[str, Any],
) -> dict[str, list[str]]:
    parent_names = {spec.name for spec in load_parameter_specs(config, parent_workspace)}
    child_names = {spec.name for spec in load_parameter_specs(config, child_workspace)}
    removed = sorted(parent_names - child_names)
    added = sorted(child_names - parent_names)
    if removed:
        raise ValueError(
            "Child calibration manifest removed inherited active parameter(s): "
            + ", ".join(removed)
        )
    maximum_new = int(
        config.get("calibration", {}).get("max_new_parameters_per_candidate", 3)
    )
    if len(added) > maximum_new:
        raise ValueError(
            f"Child calibration manifest adds {len(added)} parameters; "
            f"configured per-candidate maximum is {maximum_new}: {', '.join(added)}"
        )
    return {"inherited": sorted(parent_names), "added": added}


def _spec_from_manifest_entry(entry: dict[str, Any], path: Path) -> ParameterSpec:
    name = str(entry.get("name", "")).strip()
    if not isinstance(entry.get("calibrate", True), bool):
        raise ValueError(f"Parameter {name or '<unnamed>'} calibrate must be true or false in {path}")
    if not str(entry.get("introduced_by", "")).strip():
        raise ValueError(f"Parameter {name or '<unnamed>'} needs introduced_by metadata in {path}")
    if not str(entry.get("rationale", "")).strip():
        raise ValueError(f"Parameter {name or '<unnamed>'} needs a calibration rationale in {path}")
    bounds = entry.get("bounds")
    target = entry.get("target")
    if not isinstance(bounds, dict) or not isinstance(target, dict):
        raise ValueError(f"Parameter {name or '<unnamed>'} needs bounds and target mappings in {path}")
    try:
        index = int(target["index"]) if target.get("index") is not None else None
        return ParameterSpec(
            name=name,
            default=float(entry["default"]),
            minimum=float(bounds["min"]),
            maximum=float(bounds["max"]),
            mutation_size=float(entry["mutation_size"]),
            target_type=str(target.get("type", "")),
            file=str(target.get("file", "")),
            index=index,
            target_name=str(target["name"]) if target.get("name") is not None else None,
            data_type=str(entry.get("data_type", "float")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid calibration parameter {name or '<unnamed>'} in {path}: {exc}") from exc


def _validate_specs(specs: list[ParameterSpec], config: dict[str, Any]) -> None:
    max_parameters = config.get("calibration", {}).get("max_active_parameters")
    if max_parameters is not None and len(specs) > int(max_parameters):
        raise ValueError(
            f"Calibration manifest activates {len(specs)} parameters; configured maximum is {int(max_parameters)}."
        )
    names: set[str] = set()
    for spec in specs:
        if not PARAMETER_NAME_PATTERN.fullmatch(spec.name):
            raise ValueError(f"Invalid calibration parameter name: {spec.name!r}")
        if spec.name in names:
            raise ValueError(f"Duplicate calibration parameter name: {spec.name}")
        names.add(spec.name)
        if spec.target_type not in SUPPORTED_TARGET_TYPES:
            raise ValueError(f"Unsupported target type for {spec.name}: {spec.target_type}")
        if spec.data_type not in SUPPORTED_DATA_TYPES:
            raise ValueError(f"Unsupported data_type for {spec.name}: {spec.data_type}")
        relative = Path(spec.file)
        if not spec.file or relative.is_absolute() or ".." in relative.parts or relative.suffix != ".py":
            raise ValueError(f"Parameter {spec.name} must target a relative Python file")
        numbers = (spec.default, spec.minimum, spec.maximum, spec.mutation_size)
        if not all(math.isfinite(value) for value in numbers):
            raise ValueError(f"Parameter {spec.name} contains a non-finite numeric setting")
        if spec.minimum >= spec.maximum:
            raise ValueError(f"Parameter {spec.name} requires min < max")
        if not spec.minimum <= spec.default <= spec.maximum:
            raise ValueError(f"Default for {spec.name} must be within its bounds")
        if spec.mutation_size <= 0 or spec.mutation_size > spec.maximum - spec.minimum:
            raise ValueError(f"Mutation size for {spec.name} must be positive and no larger than its range")
        if spec.target_type == "single_params_index" and (spec.index is None or spec.index < 0):
            raise ValueError(f"Parameter {spec.name} requires a non-negative target index")
        if spec.target_type == "module_constant" and not (
            spec.target_name and PARAMETER_NAME_PATTERN.fullmatch(spec.target_name)
        ):
            raise ValueError(f"Parameter {spec.name} requires a valid module-constant target name")
        if spec.data_type == "int":
            integer_values = (spec.default, spec.minimum, spec.maximum, spec.mutation_size)
            if any(not float(value).is_integer() for value in integer_values):
                raise ValueError(f"Integer parameter {spec.name} requires integer-valued settings")


def _manifest_entry(spec: ParameterSpec) -> dict[str, Any]:
    target: dict[str, Any] = {"type": spec.target_type, "file": spec.file}
    if spec.index is not None:
        target["index"] = spec.index
    if spec.target_name is not None:
        target["name"] = spec.target_name
    convert = int if spec.data_type == "int" else float
    return {
        "name": spec.name,
        "calibrate": True,
        "data_type": spec.data_type,
        "default": convert(spec.default),
        "bounds": {"min": convert(spec.minimum), "max": convert(spec.maximum)},
        "mutation_size": convert(spec.mutation_size),
        "target": target,
        "introduced_by": "baseline",
        "rationale": "Existing Social Force Model calibration parameter.",
    }

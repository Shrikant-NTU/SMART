from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class HistoryManager:
    def __init__(self, history_dir: Path):
        self.history_dir = history_dir
        self.decimal_places = 2
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.history_dir / "history.json"
        self.pool_state_path = self.history_dir / "pool_state.json"
        self.report_path = self.history_dir / "report.md"
        self.llm_dir = self.history_dir / "llm_logs"
        self.llm_dir.mkdir(parents=True, exist_ok=True)
        self._llm_exchange_lock = threading.Lock()

    def load(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return json.loads(self.history_path.read_text(encoding="utf-8"))

    def recent(self, limit: int = 5) -> list[dict[str, Any]]:
        return [self._round_floats(self._strip_score_fields(item)) for item in self.load()[-limit:]]

    def append(self, entry: dict[str, Any]) -> None:
        entry = self._normalize_metric_containers(entry)
        history = self.load()
        history.append(self._round_floats(entry))
        self.history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        self.write_report(history)

    def load_pool_state(self) -> dict[str, Any] | None:
        if not self.pool_state_path.exists():
            return None
        return json.loads(self.pool_state_path.read_text(encoding="utf-8"))

    def save_pool_state(self, state: dict[str, Any]) -> None:
        state = self._normalize_metric_containers(state)
        self.pool_state_path.write_text(
            json.dumps(self._round_floats(state), indent=2),
            encoding="utf-8",
        )

    def save_llm_exchange(
        self,
        iteration: int,
        kind: str,
        prompt: str,
        response: str,
    ) -> dict[str, str]:
        with self._llm_exchange_lock:
            stem = f"iteration_{iteration:03d}_{kind}"
            prompt_path = self.llm_dir / f"{stem}_prompt.txt"
            response_path = self.llm_dir / f"{stem}_response.txt"
            if prompt_path.exists() or response_path.exists():
                index = 2
                while True:
                    candidate_stem = f"{stem}_{index:02d}"
                    prompt_path = self.llm_dir / f"{candidate_stem}_prompt.txt"
                    response_path = self.llm_dir / f"{candidate_stem}_response.txt"
                    if not prompt_path.exists() and not response_path.exists():
                        break
                    index += 1
            prompt_path.write_text(prompt, encoding="utf-8", errors="replace")
            response_path.write_text(response, encoding="utf-8", errors="replace")
        return {"prompt_path": str(prompt_path), "response_path": str(response_path)}

    def write_report(self, history: list[dict[str, Any]]) -> None:
        history = [self._normalize_metric_containers(item) for item in history]
        lines = [
            "# SFM Refinement Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
        ]
        for item in history:
            if item.get("mode") == "pool_refinement":
                lines.extend(self._format_pool_iteration(item))
                continue
            before = item.get("metrics_before", {})
            after = item.get("metrics_after", {})
            comparison = item.get("rule_based_comparison") or item.get("llm_metric_comparison", {})
            notes = self._clean_legacy_note(str(item.get("notes", "")))
            lines.extend(
                [
                    f"## Iteration {item.get('iteration')} - {item.get('decision')}",
                    "",
                    f"- Examples: {item.get('examples')}",
                    f"- Runtime seconds: {self._format_value(item.get('runtime_seconds'))}",
                    f"- Diagnosis: {item.get('llm_diagnosis', '')}",
                    f"- Selected candidate: {item.get('selected_candidate', {}).get('name', '')}",
                    f"- Acceptance judgement: {comparison.get('metric_judgement', comparison.get('reason', ''))}",
                    f"- Notes: {notes}",
                    "",
                    "Aggregate metrics before:",
                    "",
                    self._format_metrics(before),
                    "",
                    "Per-scenario metrics before:",
                    "",
                    self._format_scenario_metrics(before),
                    "",
                    "Aggregate metrics after:",
                    "",
                    self._format_metrics(after),
                    "",
                    "Per-scenario metrics after:",
                    "",
                    self._format_scenario_metrics(after),
                    "",
                    "Per-scenario judgement:",
                    "",
                    self._format_per_scenario_judgement(comparison),
                    "",
                    "Complexity before:",
                    "",
                    self._format_complexity(item.get("complexity_before")),
                    "",
                    "Complexity after:",
                    "",
                    self._format_complexity(item.get("complexity_after")),
                    "",
                ]
            )
        self.report_path.write_text("\n".join(lines), encoding="utf-8")

    def _format_pool_iteration(self, item: dict[str, Any]) -> list[str]:
        lines = [
            f"## Iteration {item.get('iteration')} - pool refinement",
            "",
            f"- Examples: {item.get('examples')}",
            f"- Diagnosis: {item.get('llm_diagnosis', '')}",
            f"- Notes: {item.get('notes', '')}",
            "",
            "Active pool before:",
            "",
            self._format_pool_items(item.get("pool_before", [])),
            "",
            "Children created:",
            "",
            self._format_pool_items(item.get("children_created", [])),
            "",
            "Active pool after:",
            "",
            self._format_pool_items([
                candidate
                for candidate in item.get("pool_after", [])
                if candidate.get("status") == "active"
            ]),
            "",
            "Archived children:",
            "",
            self._format_archives(item.get("archived_children", [])),
            "",
            "Literature mapping:",
            "",
            self._format_literature_mappings(item.get("literature_mappings", [])),
            "",
        ]
        return lines

    def _format_pool_items(self, items: list[dict[str, Any]]) -> str:
        if not items:
            return "- None."
        lines: list[str] = []
        for item in items:
            metrics = self._metrics_with_workspace_fallback(item)
            metric_text = ", ".join(
                f"{key}={self._format_value(value)}"
                for key, value in metrics.items()
            )
            lines.append(
                "- "
                f"{item.get('candidate_id')} "
                f"(parent={item.get('parent_id')}, "
                f"status={item.get('status')}, "
                f"expanded={item.get('has_produced_child')}): "
                f"{metric_text or 'no metrics'}"
                f"{self._domination_suffix(item)}"
            )
        return "\n".join(lines)

    def _metrics_with_workspace_fallback(self, item: dict[str, Any]) -> dict[str, Any]:
        metrics = dict(item.get("metrics", {}) or {})
        workspace = item.get("workspace")
        if not workspace:
            return metrics
        raw_metrics = self._read_workspace_metrics(Path(workspace))
        if not raw_metrics:
            return metrics

        if (metrics.get("goal_reach_rate") in {None, 0, 0.0}) and raw_metrics.get("goal_reach_rate") not in {None, 0, 0.0}:
            metrics["goal_reach_rate"] = raw_metrics["goal_reach_rate"]
        for key in (
            "average_completion_time",
            "unique_overlap_count",
            "average_overlap_duration_seconds",
            "standard_cyclomatic_complexity",
            "halstead_effort",
        ):
            if metrics.get(key) is None and raw_metrics.get(key) is not None:
                metrics[key] = raw_metrics[key]
        return metrics

    @staticmethod
    def _read_workspace_metrics(workspace: Path) -> dict[str, Any]:
        metrics_dir = workspace / ".images"
        if not metrics_dir.is_dir():
            return {}
        for path in sorted(metrics_dir.glob("*_metrics.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            scenario_metrics = raw.get("scenario_metrics")
            if isinstance(scenario_metrics, list) and scenario_metrics and isinstance(scenario_metrics[0], dict):
                merged = dict(raw)
                merged.update(scenario_metrics[0])
                return merged
            if isinstance(raw, dict):
                return raw
        return {}

    @staticmethod
    def _domination_suffix(item: dict[str, Any]) -> str:
        if item.get("status") != "dominated":
            return ""
        return (
            f" | dominated_order={item.get('dominated_order')}, "
            f"dominated_by={item.get('dominated_by')}"
        )

    def _format_archives(self, items: list[dict[str, Any]]) -> str:
        if not items:
            return "- None."
        return "\n".join(
            f"- {item.get('candidate_id')}: {item.get('status')} -> {item.get('archive_path')}"
            for item in items
        )

    def _format_literature_mappings(self, items: list[dict[str, Any]]) -> str:
        if not items:
            return "- No literature mappings recorded."
        lines: list[str] = []
        for item in items:
            target = item.get("candidate_id") or item.get("parent_candidate_id") or item.get("kind", "mapping")
            source = item.get("source") or "source not specified"
            concept = (
                item.get("relevant_concept")
                or item.get("diagnosis_or_cause")
                or ""
            )
            code_mapping = item.get("how_it_maps_to_this_code_change") or ""
            confidence = item.get("confidence") or "n/a"
            reason = item.get("confidence_reason") or "No confidence reason recorded."
            mapping_text = f"; code_mapping={code_mapping}" if code_mapping else ""
            lines.append(
                f"- {item.get('kind', 'mapping')} for {target}: {source} "
                f"({item.get('source_type', 'source')}); "
                f"concept={concept}{mapping_text}; confidence={confidence}; reason={reason}"
            )
        return "\n".join(lines)

    @staticmethod
    def _clean_legacy_note(note: str) -> str:
        if "score improved" in note or "score " in note:
            return "Legacy run recorded before metrics-only acceptance was enabled."
        return note

    def _strip_score_fields(self, value: Any) -> Any:
        if isinstance(value, dict):
            cleaned: dict[str, Any] = {}
            for key, nested in value.items():
                if key in {"score", "old_score", "new_score", "placeholder_score", "complexity_penalty"}:
                    continue
                if key == "priority_score":
                    continue
                if key == "notes" and isinstance(nested, str):
                    cleaned[key] = self._clean_legacy_note(nested)
                else:
                    cleaned[key] = self._strip_score_fields(nested)
            return cleaned
        if isinstance(value, list):
            return [self._strip_score_fields(item) for item in value]
        return value

    def _normalize_metric_containers(self, value: Any) -> Any:
        if isinstance(value, dict):
            normalized = {
                key: self._normalize_metric_containers(nested)
                for key, nested in value.items()
            }
            return self._normalize_metrics(normalized)
        if isinstance(value, list):
            return [self._normalize_metric_containers(item) for item in value]
        return value

    @staticmethod
    def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(metrics)
        if normalized.get("goal_reach_rate") not in {None, 0, 0.0}:
            return normalized
        scenario_metrics = normalized.get("scenario_metrics") or []
        weighted = 0.0
        total_weight = 0.0
        for item in scenario_metrics:
            if not isinstance(item, dict):
                continue
            rate = item.get("goal_reach_rate")
            if rate in {None, 0, 0.0}:
                continue
            weight = item.get("total_pedestrians") or 1.0
            try:
                weighted += float(rate) * float(weight)
                total_weight += float(weight)
            except (TypeError, ValueError):
                continue
        if total_weight:
            normalized["goal_reach_rate"] = weighted / total_weight
        return normalized

    def _format_metrics(self, metrics: dict[str, Any]) -> str:
        metrics = self._normalize_metrics(metrics)
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
            "lines_of_code_by_file",
        ]
        lines: list[str] = []
        for key in keys:
            if key not in metrics:
                continue
            value = metrics.get(key)
            if isinstance(value, dict):
                lines.append(f"- {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  - {sub_key}: {self._format_value(sub_value)}")
            else:
                lines.append(f"- {key}: {self._format_value(value)}")
        return "\n".join(lines) if lines else "- No metrics recorded."

    def _format_scenario_metrics(self, metrics: dict[str, Any]) -> str:
        metrics = self._normalize_metrics(metrics)
        scenario_metrics = metrics.get("scenario_metrics") or []
        if not scenario_metrics:
            return "- No per-scenario metrics recorded."

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
            "total_pedestrians",
            "reached_pedestrian_count",
            "goal_reach_definition",
            "goal_reach_condition",
        ]
        lines: list[str] = []
        for item in scenario_metrics:
            scenario_name = item.get("scenario_name", "unknown_scenario")
            lines.append(f"### {scenario_name}")
            for key in keys:
                if key in item:
                    lines.append(f"- {key}: {self._format_value(item.get(key))}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _format_per_scenario_judgement(self, comparison: dict[str, Any]) -> str:
        judgements = comparison.get("per_scenario_judgements") or []
        if not judgements:
            return "- No per-scenario judgement recorded."
        lines: list[str] = []
        for item in judgements:
            scenario_name = item.get("scenario_name", "unknown_scenario")
            judgement = item.get("judgement", "unknown")
            reason = item.get("reason", "")
            lines.append(f"- {scenario_name}: {judgement} - {reason}")
        return "\n".join(lines)

    def _format_complexity(self, snapshot: dict[str, Any] | None) -> str:
        if not snapshot:
            return "- No complexity snapshot recorded."
        metrics = snapshot.get("metrics") or {}
        if not metrics:
            errors = snapshot.get("errors") or []
            return f"- Complexity unavailable. Errors: {errors}"
        lines = []
        for key, value in metrics.items():
            lines.append(f"- {key}: {self._format_value(value)}")
        return "\n".join(lines)

    def _format_value(self, value: Any) -> str:
        if isinstance(value, bool) or value is None:
            return str(value)
        if isinstance(value, float):
            return f"{value:.{self.decimal_places}f}"
        return str(value)

    def _round_floats(self, value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float):
            return round(value, self.decimal_places)
        if isinstance(value, dict):
            return {key: self._round_floats(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [self._round_floats(item) for item in value]
        return value

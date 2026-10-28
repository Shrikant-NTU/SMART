from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class Evaluator:
    """Computes raw simulation metrics for the refinement loop.

    The trajectory parser is deliberately conservative because the exact CSV
    format is not fixed yet. Replace or extend `_metrics_from_trajectory` once
    your trajectory export has stable columns.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.decimal_places = int(config.get("metric_decimal_places", 2))

    def evaluate(
        self,
        run_result: dict[str, Any],
        workspace: Path,
        baseline_lines_of_code: int | None = None,
    ) -> dict[str, Any]:
        output_dir = Path(run_result["output_dir"])
        trajectory_path = output_dir / self.config.get("trajectory_file_name", "trajectories.csv")
        if not trajectory_path.exists():
            trajectory_path = workspace / self.config.get("trajectory_file_name", "trajectories.csv")

        metrics = self._default_metrics(run_result, workspace, baseline_lines_of_code)
        output_metrics = self._metrics_from_output_json(output_dir)
        if output_metrics:
            metrics.update(output_metrics)
        if trajectory_path.exists():
            metrics.update(self._metrics_from_trajectory(trajectory_path))

        self._normalize_aggregate_metrics(metrics)
        metrics = self._round_numbers(metrics)
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )
        return metrics

    def count_lines_of_code(self, workspace: Path) -> int:
        return sum(self.count_lines_by_file(workspace).values())

    def count_lines_by_file(self, workspace: Path) -> dict[str, int]:
        counts: dict[str, int] = {}
        for relative_name in self.config.get("line_count_files", []):
            path = workspace / relative_name
            counts[relative_name] = self._count_file_lines(path)
        return counts

    def _metrics_from_output_json(self, output_dir: Path) -> dict[str, Any]:
        metrics_dir = output_dir / "images"
        paths = sorted(metrics_dir.glob("*_metrics.json"))
        if not paths:
            return {}

        scenario_metrics: list[dict[str, Any]] = []
        for path in paths:
            try:
                scenario_metrics.append(
                    self._normalize_overlap_metrics(json.loads(path.read_text(encoding="utf-8")))
                )
            except (OSError, json.JSONDecodeError):
                continue
        if not scenario_metrics:
            return {}

        aggregated = self._aggregate_scenario_metrics(scenario_metrics)
        per_example_times = self._per_example_execution_times(scenario_metrics)
        if per_example_times:
            aggregated["per_example_execution_time_seconds"] = per_example_times
        aggregated.update(
            {
                "scenario_metrics": scenario_metrics,
                "metric_source": "per_example_json",
            }
        )
        return aggregated

    @staticmethod
    def _normalize_overlap_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
        if "unique_overlap_count" not in metrics and "overlap_count" in metrics:
            metrics["unique_overlap_count"] = metrics.get("overlap_count")
        if "overlap_episode_count" not in metrics and "unique_overlap_count" in metrics:
            metrics["overlap_episode_count"] = metrics.get("unique_overlap_count")
        metrics.setdefault("total_overlap_duration_seconds", None)
        metrics.setdefault("average_overlap_duration_seconds", None)
        metrics.setdefault("average_overlap_duration_per_pair_seconds", None)
        return metrics

    @staticmethod
    def _normalize_aggregate_metrics(metrics: dict[str, Any]) -> None:
        """Keep aggregate goal reach rate consistent with per-scenario metrics."""
        if metrics.get("goal_reach_rate") not in {None, 0, 0.0}:
            return
        scenario_metrics = metrics.get("scenario_metrics") or []
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
            metrics["goal_reach_rate"] = weighted / total_weight

    def _aggregate_scenario_metrics(self, scenario_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        aggregation_config = self.config.get("metric_aggregation", {})
        if not aggregation_config:
            return {}

        aggregated: dict[str, Any] = {}
        for metric_name, rule in aggregation_config.items():
            method = rule.get("method", "mean")
            if method == "sum":
                if metric_name == "reached_pedestrian_count":
                    derived = self._derive_reached_pedestrian_count(scenario_metrics)
                    aggregated[metric_name] = (
                        derived
                        if derived is not None
                        else self._sum_metric(scenario_metrics, metric_name)
                    )
                else:
                    aggregated[metric_name] = self._sum_metric(scenario_metrics, metric_name)
            elif method == "mean":
                aggregated[metric_name] = self._mean_metric(scenario_metrics, metric_name)
            elif method == "weighted_mean":
                aggregated[metric_name] = self._weighted_mean(
                    scenario_metrics,
                    metric_name,
                    rule.get("weight", "total_pedestrians"),
                    rule.get("default"),
                )
            elif method == "ratio":
                numerator = aggregated.get(rule.get("numerator"))
                if numerator is None:
                    numerator = self._sum_metric(scenario_metrics, rule.get("numerator", ""))
                denominator = aggregated.get(rule.get("denominator"))
                if denominator is None:
                    denominator = self._sum_metric(scenario_metrics, rule.get("denominator", ""))
                if denominator and numerator:
                    aggregated[metric_name] = numerator / denominator
                elif self._has_metric_values(scenario_metrics, metric_name):
                    aggregated[metric_name] = self._weighted_mean(
                        scenario_metrics,
                        metric_name,
                        rule.get("denominator", "total_pedestrians"),
                    )
                else:
                    aggregated[metric_name] = 0.0
            elif method == "min":
                aggregated[metric_name] = self._extreme_metric(scenario_metrics, metric_name, min)
            elif method == "max":
                aggregated[metric_name] = self._extreme_metric(scenario_metrics, metric_name, max)
        return aggregated

    def _default_metrics(
        self,
        run_result: dict[str, Any],
        workspace: Path,
        baseline_lines_of_code: int | None,
    ) -> dict[str, Any]:
        output_dir = Path(run_result["output_dir"])
        output_files = [path for path in (output_dir / "images").glob("*") if path.is_file()]
        png_count = sum(1 for path in output_files if path.suffix.lower() == ".png")
        mp4_count = sum(1 for path in output_files if path.suffix.lower() == ".mp4")
        success = not run_result.get("crashed") and not run_result.get("timed_out")
        lines_by_file = self.count_lines_by_file(workspace)
        lines = sum(lines_by_file.values())
        baseline = baseline_lines_of_code if baseline_lines_of_code is not None else lines
        execution_time_seconds = run_result.get("runtime_seconds", 0.0)
        failure = self._simulation_failure_summary(run_result) if not success else None

        overlap_default: float | int | None = 0.0 if success else None
        count_default: int | None = 0 if success else None

        metrics = {
            "goal_reach_rate": 0.5 if success else 0.0,
            "completion_time": execution_time_seconds,
            "stuck_pedestrian_count": count_default,
            "unique_overlap_count": count_default,
            "overlap_episode_count": count_default,
            "total_overlap_duration_seconds": overlap_default,
            "average_overlap_duration_seconds": overlap_default,
            "average_overlap_duration_per_pair_seconds": overlap_default,
            "overlap_count": count_default,
            "average_speed": None,
            "execution_time": execution_time_seconds,
            "execution_time_seconds": execution_time_seconds,
            "lines_of_code_by_file": lines_by_file,
            "lines_of_code": lines,
            "lines_added_vs_baseline": lines - baseline,
            "png_count": png_count,
            "mp4_count": mp4_count,
            "simulation_success": success,
        }
        if failure:
            metrics["simulation_failure"] = failure
        return metrics

    @staticmethod
    def _simulation_failure_summary(run_result: dict[str, Any]) -> dict[str, Any]:
        script_runs = run_result.get("script_runs") or []
        failed = next(
            (
                item
                for item in script_runs
                if item.get("returncode") not in (None, 0) or item.get("timed_out")
            ),
            {},
        )
        return {
            "crashed": bool(run_result.get("crashed")),
            "timed_out": bool(run_result.get("timed_out")),
            "returncode": failed.get("returncode", run_result.get("returncode")),
            "example_number": failed.get("example_number"),
            "script_name": failed.get("script_name"),
            "stderr_path": failed.get("stderr_path"),
            "error": failed.get("error", ""),
        }

    def _metrics_from_trajectory(self, path: Path) -> dict[str, Any]:
        rows: list[dict[str, str]] = []
        with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

        if not rows:
            return {}

        columns = set(rows[0])
        metrics: dict[str, Any] = {"trajectory_rows": len(rows)}

        if {"time", "pedestrian_id", "x", "y"}.issubset(columns):
            metrics.update(self._basic_speed_metrics(rows))

        return metrics

    def _basic_speed_metrics(self, rows: list[dict[str, str]]) -> dict[str, Any]:
        by_id: dict[str, list[tuple[float, float, float]]] = {}
        for row in rows:
            try:
                pid = row["pedestrian_id"]
                by_id.setdefault(pid, []).append(
                    (float(row["time"]), float(row["x"]), float(row["y"]))
                )
            except (KeyError, TypeError, ValueError):
                continue

        speeds: list[float] = []
        distances: list[float] = []
        for samples in by_id.values():
            samples.sort()
            if len(samples) < 2:
                continue
            pedestrian_distance = 0.0
            for first, second in zip(samples, samples[1:]):
                dt = max(second[0] - first[0], 1e-9)
                distance = ((first[1] - second[1]) ** 2 + (first[2] - second[2]) ** 2) ** 0.5
                pedestrian_distance += distance
                speeds.append(distance / dt)
            distances.append(pedestrian_distance)

        return {
            "average_speed": sum(speeds) / len(speeds) if speeds else None,
            "total_distance_traveled": sum(distances),
            "average_distance_traveled": sum(distances) / len(distances) if distances else None,
        }

    @staticmethod
    def _sum_metric(items: list[dict[str, Any]], value_key: str) -> float | int:
        values = [item.get(value_key) for item in items if item.get(value_key) is not None]
        total = sum((float(value) for value in values), 0.0)
        return int(total) if total.is_integer() else total

    @staticmethod
    def _has_metric_values(items: list[dict[str, Any]], value_key: str) -> bool:
        return any(item.get(value_key) is not None for item in items)

    @staticmethod
    def _derive_reached_pedestrian_count(items: list[dict[str, Any]]) -> float | int | None:
        total = 0.0
        found = False
        for item in items:
            reached = item.get("reached_pedestrian_count")
            if reached is not None:
                total += float(reached)
                found = True
                continue
            rate = item.get("goal_reach_rate")
            pedestrians = item.get("total_pedestrians")
            if rate is None or pedestrians is None:
                continue
            total += float(rate) * float(pedestrians)
            found = True
        if not found:
            return None
        return int(total) if total.is_integer() else total

    @staticmethod
    def _per_example_execution_times(items: list[dict[str, Any]]) -> dict[str, float]:
        times: dict[str, float] = {}
        for index, item in enumerate(items, start=1):
            value = item.get("execution_time_seconds")
            if value is None:
                continue
            name = item.get("scenario_name") or item.get("example_name") or f"example_{index}"
            try:
                times[str(name)] = float(value)
            except (TypeError, ValueError):
                continue
        return times

    @staticmethod
    def _mean_metric(items: list[dict[str, Any]], value_key: str) -> float | None:
        values = [float(item[value_key]) for item in items if item.get(value_key) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    @staticmethod
    def _extreme_metric(items: list[dict[str, Any]], value_key: str, func: Any) -> float | None:
        values = [float(item[value_key]) for item in items if item.get(value_key) is not None]
        if not values:
            return None
        return func(values)

    @staticmethod
    def _weighted_mean(
        items: list[dict[str, Any]],
        value_key: str,
        weight_key: str,
        default: float | None = None,
    ) -> float | None:
        numerator = 0.0
        denominator = 0.0
        for item in items:
            value = item.get(value_key)
            if value is None:
                continue
            weight = float(item.get(weight_key, 1) or 1)
            numerator += float(value) * weight
            denominator += weight
        if denominator == 0:
            return default
        return numerator / denominator

    @staticmethod
    def _count_file_lines(path: Path) -> int:
        if not path.exists():
            return 0
        return sum(
            1
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )

    def _round_numbers(self, value: Any) -> Any:
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, float):
            return round(value, self.decimal_places)
        if isinstance(value, dict):
            return {key: self._round_numbers(nested) for key, nested in value.items()}
        if isinstance(value, list):
            return [self._round_numbers(item) for item in value]
        return value

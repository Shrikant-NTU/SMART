from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def compute_simulation_metrics(
    all_states: np.ndarray,
    humans_goal: np.ndarray,
    humans_parameters: np.ndarray,
    dt: float,
    goal_tolerance: float | None = None,
    reached_mask: np.ndarray | None = None,
    stuck_speed_threshold: float = 0.05,
    stuck_duration_seconds: float = 5.0,
    overlap_mode: str = "full",
) -> dict[str, Any]:
    states = np.asarray(all_states, dtype=float)
    goals = np.asarray(humans_goal, dtype=float)
    parameters = np.asarray(humans_parameters, dtype=float)
    positions = states[:, :, :2]
    radii = parameters[:, 0] if parameters.ndim == 2 and parameters.shape[1] > 0 else np.full(states.shape[1], 0.25)
    tolerance = float(goal_tolerance) if goal_tolerance is not None else float(max(0.30, np.nanmean(radii) * 1.5))

    distances_to_goal = np.linalg.norm(positions - goals[None, :, :], axis=2)
    if reached_mask is None:
        reached_mask = distances_to_goal <= tolerance
        goal_reach_definition = "distance_to_goal"
    else:
        reached_mask = np.asarray(reached_mask, dtype=bool)
        if reached_mask.shape != distances_to_goal.shape:
            raise ValueError(
                "reached_mask must have shape "
                f"{distances_to_goal.shape}, got {reached_mask.shape}"
            )
        goal_reach_definition = "scenario_specific_reached_mask"
    ever_reached = np.any(reached_mask, axis=0)
    first_reach_indices = np.argmax(reached_mask, axis=0)
    completion_times = np.where(ever_reached, first_reach_indices * dt, np.nan)

    if overlap_mode == "calibration_basic":
        overlap_metrics = _overlap_metrics(positions, radii, dt, mode="full")
        return {
            "goal_reach_rate": float(np.mean(ever_reached)) if len(ever_reached) else 0.0,
            "average_completion_time": _nanmean_or_none(completion_times),
            "average_overlap_duration_seconds": overlap_metrics["average_overlap_duration_seconds"],
        }

    speeds = _estimate_speeds(states, positions, dt)
    stuck_agents = _count_stuck_agents(
        speeds,
        reached_mask,
        stuck_speed_threshold,
        stuck_duration_seconds,
        dt,
    )
    overlap_metrics = _overlap_metrics(positions, radii, dt, mode=overlap_mode)
    if overlap_mode == "total_only":
        return {"overlap_count": overlap_metrics["overlap_count"]}

    return {
        "goal_reach_rate": float(np.mean(ever_reached)) if len(ever_reached) else 0.0,
        "total_pedestrians": int(states.shape[1]),
        "average_completion_time": _nanmean_or_none(completion_times),
        "average_speed": float(np.nanmean(speeds)) if speeds.size else None,
        "stuck_pedestrian_count": int(stuck_agents),
        "overlap_count": overlap_metrics["overlap_count"],
        "unique_overlap_count": overlap_metrics["unique_overlap_count"],
        "overlap_episode_count": overlap_metrics["overlap_episode_count"],
        "total_overlap_duration_seconds": overlap_metrics["total_overlap_duration_seconds"],
        "average_overlap_duration_seconds": overlap_metrics["average_overlap_duration_seconds"],
    }


def save_simulation_metrics(
    all_states: np.ndarray,
    humans_goal: np.ndarray,
    humans_parameters: np.ndarray,
    dt: float,
    output_path: str | Path,
    scenario_name: str,
    reached_mask: np.ndarray | None = None,
    extra: dict[str, Any] | None = None,
    overlap_mode: str = "full",
) -> dict[str, Any]:
    metrics = compute_simulation_metrics(
        all_states,
        humans_goal,
        humans_parameters,
        dt,
        reached_mask=reached_mask,
        overlap_mode=overlap_mode,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Metrics saved to: {path}")
    return metrics


def _estimate_speeds(states: np.ndarray, positions: np.ndarray, dt: float) -> np.ndarray:
    if states.shape[-1] >= 6:
        return np.linalg.norm(states[:, :, 4:6], axis=2)
    if states.shape[-1] >= 4:
        return np.linalg.norm(states[:, :, 2:4], axis=2)
    if len(positions) < 2:
        return np.empty((0, positions.shape[1]))
    displacements = np.diff(positions, axis=0)
    return np.linalg.norm(displacements, axis=2) / max(float(dt), 1e-12)


def _distance_traveled(positions: np.ndarray) -> dict[str, float | None]:
    if len(positions) < 2:
        return {"total_distance_traveled": 0.0, "average_distance_traveled": None}

    per_pedestrian: list[float] = []
    for pedestrian in range(positions.shape[1]):
        samples = positions[:, pedestrian, :]
        finite = np.all(np.isfinite(samples), axis=1)
        distance = 0.0
        for first, second, first_ok, second_ok in zip(
            samples[:-1],
            samples[1:],
            finite[:-1],
            finite[1:],
        ):
            if first_ok and second_ok:
                distance += float(np.linalg.norm(second - first))
        per_pedestrian.append(distance)

    total = float(np.sum(per_pedestrian))
    average = float(np.mean(per_pedestrian)) if per_pedestrian else None
    return {"total_distance_traveled": total, "average_distance_traveled": average}


def _count_stuck_agents(
    speeds: np.ndarray,
    reached_mask: np.ndarray,
    speed_threshold: float,
    duration_seconds: float,
    dt: float,
) -> int:
    if speeds.size == 0:
        return 0
    required_steps = max(1, int(round(duration_seconds / max(float(dt), 1e-12))))
    not_reached_yet = ~np.maximum.accumulate(reached_mask, axis=0)
    slow = (speeds < speed_threshold) & not_reached_yet[: speeds.shape[0]]
    stuck = 0
    for pedestrian in range(slow.shape[1]):
        run_length = 0
        for is_slow in slow[:, pedestrian]:
            run_length = run_length + 1 if is_slow else 0
            if run_length >= required_steps:
                stuck += 1
                break
    return stuck


def _overlap_metrics(positions: np.ndarray, radii: np.ndarray, dt: float, mode: str = "full") -> dict[str, float | int | None]:
    n_pedestrians = positions.shape[1]
    if n_pedestrians < 2 or positions.shape[0] == 0:
        return {
            "overlap_count": 0,
            "unique_overlap_count": 0,
            "overlap_episode_count": 0,
            "total_overlap_duration_seconds": 0.0,
            "average_overlap_duration_seconds": 0.0,
            "average_overlap_duration_per_pair_seconds": 0.0,
        }

    overlap_matrix = _pair_overlap_matrix(positions, radii)
    pair_frame_counts = np.sum(overlap_matrix, axis=0)
    frame_pair_overlaps = int(np.sum(pair_frame_counts))
    if mode == "total_only":
        return {
            "overlap_count": frame_pair_overlaps,
            "unique_overlap_count": None,
            "overlap_episode_count": None,
            "total_overlap_duration_seconds": None,
            "average_overlap_duration_seconds": None,
            "average_overlap_duration_per_pair_seconds": None,
        }

    overlapping_pair_counts = pair_frame_counts[pair_frame_counts > 0]
    pair_durations = overlapping_pair_counts.astype(float) * float(dt)
    episode_durations = _overlap_episode_durations_from_matrix(overlap_matrix, dt)
    total_duration = float(np.sum(pair_durations)) if len(pair_durations) else 0.0
    average_duration_per_pedestrian = total_duration / float(n_pedestrians)
    average_pair_duration = float(np.mean(pair_durations)) if len(pair_durations) else 0.0
    return {
        "overlap_count": frame_pair_overlaps,
        "unique_overlap_count": int(len(overlapping_pair_counts)),
        "overlap_episode_count": int(len(episode_durations)),
        "total_overlap_duration_seconds": total_duration,
        "average_overlap_duration_seconds": average_duration_per_pedestrian,
        "average_overlap_duration_per_pair_seconds": average_pair_duration,
    }


def _pair_overlap_matrix(positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
    pair_i, pair_j = np.triu_indices(positions.shape[1], k=1)
    first = positions[:, pair_i, :]
    second = positions[:, pair_j, :]
    finite_pairs = np.all(np.isfinite(first), axis=2) & np.all(np.isfinite(second), axis=2)
    distances = np.linalg.norm(first - second, axis=2)
    required_distances = radii[pair_i] + radii[pair_j]
    return (distances < required_distances[None, :]) & finite_pairs


def _overlap_episode_durations_from_matrix(overlap_matrix: np.ndarray, dt: float) -> list[float]:
    if overlap_matrix.size == 0:
        return []

    padded = np.pad(overlap_matrix.astype(np.int8), ((1, 1), (0, 0)), constant_values=0)
    changes = np.diff(padded, axis=0)
    starts = np.argwhere(changes == 1)
    ends = np.argwhere(changes == -1)
    if len(starts) == 0:
        return []

    starts = starts[np.lexsort((starts[:, 0], starts[:, 1]))]
    ends = ends[np.lexsort((ends[:, 0], ends[:, 1]))]
    run_lengths = ends[:, 0] - starts[:, 0]
    return (run_lengths.astype(float) * float(dt)).tolist()


def _nanmean_or_none(values: np.ndarray) -> float | None:
    if np.all(np.isnan(values)):
        return None
    return float(np.nanmean(values))

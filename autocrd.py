from __future__ import annotations

import argparse
import base64
import copy
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)

from openai import OpenAI

from openai_key import resolve_openai_api_key
from prompt_autocrd import (
    code_video_conceptual_model_prompt,
    code_video_face_validation_prompt,
    code_video_indicator_generation_prompt,
    code_video_insight_prompt,
    code_video_scoring_prompt,
    conceptual_model_prompt,
    conceptual_model_score_prompt,
    face_validation_prompt,
    indicator_generation_prompt,
    indicator_score_adjust_prompt,
    indicator_update_prompt,
    insight_generation_prompt,
    per_run_scoring_prompt,
    plotting_helper_prompt,
)


DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_MAX_SAMPLE_ROWS = 250
DEFAULT_GRAPH_COUNT = 6
DEFAULT_INDICATOR_COUNT = 4
DEFAULT_PLAUSIBILITY_THRESHOLD = 0.65


def make_openai_client() -> OpenAI:
    return OpenAI(api_key=resolve_openai_api_key())


def safe_json_loads(text: str) -> dict[str, Any]:
    cleaned = _extract_fenced_block(text, language="json").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM did not return valid JSON. Error: {exc}\nRaw:\n{text}") from exc


def load_timeseries_csv(
    csv_path: Path,
    run_column: str | None = None,
    parameter_columns: list[str] | None = None,
) -> list[dict[str, Any]]:
    df = pd.read_csv(csv_path)
    parameter_columns = parameter_columns or []

    if run_column is None:
        run_column = _infer_run_column(df.columns)

    runs: list[dict[str, Any]] = []
    if run_column:
        grouped = df.groupby(run_column, sort=False)
        iterator = ((run_id, group.drop(columns=[run_column])) for run_id, group in grouped)
    else:
        iterator = [(0, df)]

    for run_index, (run_id, group) in enumerate(iterator):
        params: dict[str, Any] = {}
        for column in parameter_columns:
            if column in group.columns:
                values = group[column].dropna().unique()
                if len(values) == 1:
                    params[column] = _json_scalar(values[0])
                elif len(values) > 1:
                    params[column] = [_json_scalar(value) for value in values.tolist()]

        data_columns = [column for column in group.columns if column not in parameter_columns]
        numeric_group = group[data_columns].apply(pd.to_numeric, errors="coerce")
        numeric_group = numeric_group.dropna(axis=1, how="all")
        data = {
            column: [_json_scalar(value) for value in numeric_group[column].dropna().tolist()]
            for column in numeric_group.columns
        }
        runs.append(
            {
                "run_index": run_index,
                "run_id": _json_scalar(run_id),
                "params": params,
                "data": data,
            }
        )

    return runs


def build_conceptual_model(
    client: OpenAI,
    model: str,
    simulation_description: str,
    data_columns: list[str],
    parameter_names: list[str],
) -> dict[str, Any]:
    raw = call_text(
        client,
        model,
        conceptual_model_prompt(simulation_description, data_columns, parameter_names),
    )
    conceptual_model = safe_json_loads(raw)
    conceptual_model.setdefault("expected_behaviours", [])
    return conceptual_model


def score_conceptual_model(
    client: OpenAI,
    model: str,
    conceptual_model: dict[str, Any],
) -> dict[str, Any]:
    raw = call_text(client, model, conceptual_model_score_prompt(conceptual_model))
    result = safe_json_loads(raw)
    score = float(result.get("validity_score", 0.0))
    result["validity_score"] = max(0.0, min(1.0, score))
    return result


def build_code_video_conceptual_model(
    client: OpenAI,
    model: str,
    simulation_description: str,
    source_context: str,
    visual_summary: dict[str, Any],
) -> dict[str, Any]:
    raw = call_text(
        client,
        model,
        code_video_conceptual_model_prompt(
            simulation_description,
            source_context,
            visual_summary,
        ),
    )
    return safe_json_loads(raw)


def generate_plot_helper(
    client: OpenAI,
    model: str,
    simulation_description: str,
    conceptual_model: dict[str, Any],
    example_data: dict[str, list[Any]],
    helper_path: Path,
    graph_count: int = DEFAULT_GRAPH_COUNT,
) -> str:
    prompt = plotting_helper_prompt(
        simulation_description,
        conceptual_model,
        example_data,
        graph_count=graph_count,
    )
    raw = call_text(client, model, prompt)
    code = sanitize_plot_helper_code(_extract_fenced_block(raw, language="python").strip())
    helper_path.write_text(code + "\n", encoding="utf-8")
    return code


def generate_plots_for_runs(
    runs: list[dict[str, Any]],
    helper_path: Path,
    plots_dir: Path,
    graph_count: int = DEFAULT_GRAPH_COUNT,
) -> list[dict[str, Any]]:
    sanitize_plot_helper_file(helper_path)
    helper_module = _load_module_from_path("autocrd_generated_plot_helper", helper_path)
    generate_plots = getattr(helper_module, "generate_plots")
    plot_records: list[dict[str, Any]] = []

    for run in runs:
        run_dir = plots_dir / f"run_{int(run['run_index']) + 1:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        generate_plots(run["data"], str(run_dir))
        image_paths = [str(run_dir / f"graph{index}.png") for index in range(1, graph_count + 1)]
        missing = [path for path in image_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(
                "Generated plot helper did not produce the expected graph files: "
                + ", ".join(missing)
            )
        plot_records.append(
            {
                "run_index": run["run_index"],
                "run_id": run.get("run_id"),
                "run_dir": str(run_dir),
                "image_paths": image_paths,
            }
        )

    return plot_records


def sanitize_plot_helper_file(helper_path: Path) -> None:
    original = helper_path.read_text(encoding="utf-8", errors="replace")
    sanitized = sanitize_plot_helper_code(original)
    if sanitized != original:
        helper_path.write_text(sanitized, encoding="utf-8")


def sanitize_plot_helper_code(code: str) -> str:
    replacements = {
        ", sharex=False": "",
        ", sharey=False": "",
        "sharex=False, ": "",
        "sharey=False, ": "",
    }
    sanitized = code
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
    return sanitized


def generate_indicators(
    client: OpenAI,
    model: str,
    simulation_description: str,
    conceptual_model: dict[str, Any],
    indicator_count: int = 4,
) -> list[dict[str, Any]]:
    raw = call_text(
        client,
        model,
        indicator_generation_prompt(simulation_description, conceptual_model, indicator_count),
    )
    result = safe_json_loads(raw)
    return normalize_indicator_weights(result["indicators"])


def generate_code_video_indicators(
    client: OpenAI,
    model: str,
    simulation_description: str,
    conceptual_model: dict[str, Any],
    indicator_count: int = DEFAULT_INDICATOR_COUNT,
) -> list[dict[str, Any]]:
    raw = call_text(
        client,
        model,
        code_video_indicator_generation_prompt(
            simulation_description,
            conceptual_model,
            indicator_count,
        ),
    )
    result = safe_json_loads(raw)
    return normalize_indicator_weights(result["indicators"])


def generate_insights_for_runs(
    client: OpenAI,
    model: str,
    runs: list[dict[str, Any]],
    plot_records: list[dict[str, Any]],
    simulation_description: str,
    conceptual_model: dict[str, Any],
) -> list[dict[str, Any]]:
    expectations = list(conceptual_model.get("expected_behaviours", []))
    run_by_index = {run["run_index"]: run for run in runs}
    insights: list[dict[str, Any]] = []

    for record in plot_records:
        run = run_by_index[record["run_index"]]
        numeric_summary = summarize_run_data(run["data"])
        prompt = insight_generation_prompt(
            run_index=int(record["run_index"]),
            run_dir=str(record["run_dir"]),
            simulation_description=simulation_description,
            conceptual_expectations=expectations,
            numeric_summary=numeric_summary,
        )
        raw = call_with_images(
            client,
            model,
            prompt,
            [Path(path) for path in record.get("image_paths", [])],
        )
        insights.append(
            {
                "run_index": record["run_index"],
                "run_id": record.get("run_id"),
                "run_dir": record["run_dir"],
                "image_paths": record.get("image_paths", []),
                "numeric_summary": numeric_summary,
                "insights": raw,
            }
        )

    return insights


def score_run_indicators(
    client: OpenAI,
    model: str,
    simulation_description: str,
    indicators: list[dict[str, Any]],
    conceptual_expectations: list[str],
    insights_per_run: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expectations_text = "\n".join(f"- {item}" for item in conceptual_expectations)
    scored_runs: list[dict[str, Any]] = []

    for run in insights_per_run:
        messages = [
            {"role": "system", "content": "You are an expert simulation evaluator assigning calibrated indicator scores."},
            {"role": "user", "content": "SIMULATION DESCRIPTION:\n" + simulation_description},
            {"role": "user", "content": "INDICATORS:\n" + json.dumps(indicators, indent=2)},
            {"role": "user", "content": "CONCEPTUAL EXPECTATIONS:\n" + expectations_text},
            {"role": "user", "content": f"RUN INDEX: {run['run_index']}\nRUN INSIGHTS:\n{run['insights']}"},
            {"role": "user", "content": per_run_scoring_prompt()},
        ]
        raw = call_messages(client, model, messages)
        per_run = safe_json_loads(raw)
        scored_runs.append(
            {
                "run_index": run["run_index"],
                "run_id": run.get("run_id"),
                "scores": per_run["scores"],
            }
        )

    return scored_runs


def generate_code_video_insights(
    client: OpenAI,
    model: str,
    simulation_description: str,
    conceptual_model: dict[str, Any],
    source_context: str,
    metrics: dict[str, Any],
    visual_summary: dict[str, Any],
    image_paths: list[Path],
) -> dict[str, Any]:
    prompt = code_video_insight_prompt(
        simulation_description,
        conceptual_model,
        source_context,
        metrics,
        visual_summary,
    )
    raw = call_with_images(client, model, prompt, image_paths)
    return {
        "image_paths": [str(path) for path in image_paths],
        "visual_summary": visual_summary,
        "metrics": metrics,
        "insights": raw,
    }


def score_code_video_indicators(
    client: OpenAI,
    model: str,
    simulation_description: str,
    indicators: list[dict[str, Any]],
    conceptual_model: dict[str, Any],
    insights: dict[str, Any],
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are a strict Social Force Model implementation-plausibility evaluator."},
        {"role": "user", "content": "SIMULATION DESCRIPTION:\n" + simulation_description},
        {"role": "user", "content": "INDICATORS:\n" + json.dumps(indicators, indent=2)},
        {"role": "user", "content": "CONCEPTUAL MODEL:\n" + json.dumps(conceptual_model, indent=2)},
        {"role": "user", "content": "CODE/VIDEO FINDINGS:\n" + str(insights.get("insights", ""))},
        {"role": "user", "content": code_video_scoring_prompt()},
    ]
    scored = safe_json_loads(call_messages(client, model, messages))
    return {
        "run_index": 0,
        "run_id": "code_video_assessment",
        "scores": scored["scores"],
    }


def compute_batch_credibility_from_run_indicators(
    run_indicator_scores: list[dict[str, Any]],
    indicators: list[dict[str, Any]],
) -> dict[str, Any]:
    name_to_values: dict[str, list[float]] = {indicator["name"]: [] for indicator in indicators}
    name_to_weight: dict[str, float] = {
        indicator["name"]: float(indicator["weight"]) for indicator in indicators
    }

    for run in run_indicator_scores:
        for score in run["scores"]:
            name = score["name"]
            if name in name_to_values:
                name_to_values[name].append(float(score["score"]))

    per_indicator_batch: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for name, values in name_to_values.items():
        weight = name_to_weight.get(name, 0.0)
        mean_score = sum(values) / len(values) if values else 0.0
        per_indicator_batch.append(
            {
                "name": name,
                "weight": weight,
                "mean_across_runs": mean_score,
                "num_runs": len(values),
            }
        )
        if values and weight > 0.0:
            weighted_sum += weight * mean_score
            weight_total += weight

    return {
        "batch_credibility": weighted_sum / weight_total if weight_total > 0.0 else 0.0,
        "per_indicator_batch": per_indicator_batch,
        "aggregator": {
            "type": "weighted_mean_of_indicator_means",
            "note": "Batch credibility is the weighted arithmetic mean of per-indicator mean scores across runs.",
        },
    }


def compute_model_plausibility_from_indicator_scores(
    model_indicator_scores: dict[str, Any],
    indicators: list[dict[str, Any]],
) -> dict[str, Any]:
    name_to_weight = {
        indicator["name"]: float(indicator["weight"]) for indicator in indicators
    }
    indicator_summary: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for score in model_indicator_scores.get("scores", []):
        name = str(score.get("name", ""))
        if name not in name_to_weight:
            continue
        weight = name_to_weight[name]
        value = max(0.0, min(1.0, float(score.get("score", 0.0))))
        indicator_summary.append(
            {
                "name": name,
                "weight": weight,
                "score": value,
                "reasoning": score.get("reasoning", ""),
            }
        )
        if weight > 0.0:
            weighted_sum += weight * value
            weight_total += weight

    return {
        "model_plausibility": weighted_sum / weight_total if weight_total > 0.0 else 0.0,
        "per_indicator_model": indicator_summary,
        "aggregator": {
            "type": "weighted_mean_of_single_assessment_indicator_scores",
            "note": "Model plausibility is the weighted arithmetic mean of indicator scores from one deterministic tuned-model assessment.",
        },
    }


def face_validate(
    client: OpenAI,
    model: str,
    simulation_description: str,
    indicators: list[dict[str, Any]],
    assessment_summary: dict[str, Any],
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are performing an expert human-style face validity check."},
        {"role": "user", "content": "SIMULATION DESCRIPTION:\n" + simulation_description},
        {"role": "user", "content": "CURRENT INDICATORS:\n" + json.dumps(indicators, indent=2)},
        {"role": "user", "content": "MODEL ASSESSMENT SUMMARY:\n" + json.dumps(assessment_summary, indent=2)},
        {"role": "user", "content": face_validation_prompt()},
    ]
    return safe_json_loads(call_messages(client, model, messages))


def face_validate_code_video(
    client: OpenAI,
    model: str,
    simulation_description: str,
    indicators: list[dict[str, Any]],
    assessment_summary: dict[str, Any],
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": "You are performing a strict expert face-validity check for SFM implementation plausibility."},
        {"role": "user", "content": "SIMULATION DESCRIPTION:\n" + simulation_description},
        {"role": "user", "content": "CURRENT INDICATORS:\n" + json.dumps(indicators, indent=2)},
        {"role": "user", "content": "MODEL ASSESSMENT SUMMARY:\n" + json.dumps(assessment_summary, indent=2)},
        {"role": "user", "content": code_video_face_validation_prompt()},
    ]
    return safe_json_loads(call_messages(client, model, messages))


def apply_indicator_score_adjustments(
    run_indicator_scores: list[dict[str, Any]],
    adjustments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    updated = copy.deepcopy(run_indicator_scores)
    index_map = {run["run_index"]: run for run in updated}

    for adjustment in adjustments:
        run_index = adjustment["run_index"]
        indicator_name = adjustment["indicator_name"]
        new_score = max(0.0, min(1.0, float(adjustment["new_score"])))
        reason = adjustment.get("reason", "")
        if run_index not in index_map:
            continue
        for score in index_map[run_index]["scores"]:
            if score["name"] == indicator_name:
                score["score"] = new_score
                score["reasoning"] = (
                    score.get("reasoning", "").strip() + " | ADJUSTED: " + reason
                ).strip()
                break

    return updated


def request_indicator_score_adjustments(
    client: OpenAI,
    model: str,
    simulation_description: str,
    indicators: list[dict[str, Any]],
    assessment_summary: dict[str, Any],
    target_issue: dict[str, Any],
) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": "You adjust simulation plausibility scores conservatively."},
        {"role": "user", "content": "SIMULATION DESCRIPTION:\n" + simulation_description},
        {"role": "user", "content": "CURRENT INDICATORS:\n" + json.dumps(indicators, indent=2)},
        {"role": "user", "content": "MODEL ASSESSMENT SUMMARY:\n" + json.dumps(assessment_summary, indent=2)},
        {"role": "user", "content": "TARGET ISSUE:\n" + json.dumps(target_issue, indent=2)},
        {"role": "user", "content": indicator_score_adjust_prompt()},
    ]
    return safe_json_loads(call_messages(client, model, messages)).get("adjustments", [])


def request_indicator_update(
    client: OpenAI,
    model: str,
    simulation_description: str,
    indicators: list[dict[str, Any]],
    face_validation: dict[str, Any],
) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": "You update transparent, reusable indicators with minimal drift."},
        {"role": "user", "content": "SIMULATION DESCRIPTION:\n" + simulation_description},
        {"role": "user", "content": "CURRENT INDICATORS:\n" + json.dumps(indicators, indent=2)},
        {"role": "user", "content": "FACE VALIDATION OUTPUT:\n" + json.dumps(face_validation, indent=2)},
        {"role": "user", "content": indicator_update_prompt()},
    ]
    result = safe_json_loads(call_messages(client, model, messages))
    updated = [
        {
            "name": item["name"],
            "description": item["description"],
            "weight": float(item["weight"]),
        }
        for item in result["indicators"]
    ]
    return normalize_indicator_weights(updated)


def bootstrap_autocred_code_video_rubric(
    simulation_description: str,
    source_context: str,
    visual_summary: dict[str, Any],
    image_paths: list[Path],
    state_dir: Path,
    model: str = DEFAULT_MODEL,
    indicator_count: int = DEFAULT_INDICATOR_COUNT,
) -> dict[str, Any]:
    """Create and save the fixed AutoCred rubric from optimized base code/video evidence."""
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    client = make_openai_client()
    conceptual_model = build_code_video_conceptual_model(
        client,
        model,
        simulation_description,
        source_context,
        visual_summary,
    )
    indicators = generate_code_video_indicators(
        client,
        model,
        simulation_description,
        conceptual_model,
        indicator_count,
    )
    state = {
        "schema_version": 1,
        "mode": "code_video",
        "model": model,
        "indicator_count": indicator_count,
        "simulation_description": simulation_description,
        "conceptual_model": conceptual_model,
        "indicators": indicators,
        "bootstrap_visual_summary": visual_summary,
        "bootstrap_image_paths": [str(path) for path in image_paths],
    }
    save_autocred_state(state, state_dir / "autocrd_code_video_state.json")
    write_json(state_dir / "code_video_conceptual_model.json", conceptual_model)
    write_json(state_dir / "code_video_indicators.json", {"indicators": indicators})
    (state_dir / "code_video_bootstrap_source_context.txt").write_text(
        source_context,
        encoding="utf-8",
        errors="replace",
    )
    return state


def bootstrap_autocred_rubric(
    csv_path: Path,
    simulation_description: str,
    state_dir: Path,
    model: str = DEFAULT_MODEL,
    run_column: str | None = None,
    parameter_columns: list[str] | None = None,
    indicator_count: int = DEFAULT_INDICATOR_COUNT,
    graph_count: int = DEFAULT_GRAPH_COUNT,
    max_sample_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
) -> dict[str, Any]:
    """Create and save the fixed AutoCred rubric from the optimized base run."""
    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    client = make_openai_client()
    runs = load_timeseries_csv(csv_path, run_column, parameter_columns)
    if not runs:
        raise ValueError(f"No runs found in {csv_path}")
    require_single_model_assessment(runs, csv_path)

    first_data = runs[0]["data"]
    data_columns = list(first_data.keys())
    parameter_names = sorted({name for run in runs for name in run.get("params", {}).keys()})
    conceptual_model = build_conceptual_model(
        client,
        model,
        simulation_description,
        data_columns,
        parameter_names,
    )
    conceptual_evaluation = score_conceptual_model(client, model, conceptual_model)
    helper_path = (state_dir / "generated_plot_helper.py").resolve()
    generate_plot_helper(
        client,
        model,
        simulation_description,
        conceptual_model,
        sample_run_data(first_data, max_sample_rows),
        helper_path,
        graph_count=graph_count,
    )
    indicators = generate_indicators(
        client,
        model,
        simulation_description,
        conceptual_model,
        indicator_count,
    )
    state = {
        "schema_version": 1,
        "source_csv": str(csv_path),
        "model": model,
        "run_column": run_column,
        "parameter_columns": parameter_columns or [],
        "indicator_count": indicator_count,
        "graph_count": graph_count,
        "max_sample_rows": max_sample_rows,
        "simulation_description": simulation_description,
        "conceptual_model": conceptual_model,
        "conceptual_evaluation": conceptual_evaluation,
        "indicators": indicators,
        "plot_helper_path": str(helper_path),
    }
    save_autocred_state(state, state_dir / "autocrd_state.json")
    write_json(state_dir / "conceptual_model.json", conceptual_model)
    write_json(state_dir / "indicators.json", {"indicators": indicators})
    return state


def score_code_video_plausibility_with_fixed_rubric(
    source_context: str,
    visual_summary: dict[str, Any],
    metrics: dict[str, Any],
    image_paths: list[Path],
    state: dict[str, Any],
    out_dir: Path,
    model: str | None = None,
    plausibility_threshold: float = DEFAULT_PLAUSIBILITY_THRESHOLD,
    run_face_validation: bool = False,
) -> dict[str, Any]:
    """Score one candidate from complete source files and video frames."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_openai_client()
    model_name = model or str(state.get("model") or DEFAULT_MODEL)
    simulation_description = str(state["simulation_description"])
    conceptual_model = dict(state["conceptual_model"])
    indicators = list(state["indicators"])

    insights = generate_code_video_insights(
        client,
        model_name,
        simulation_description,
        conceptual_model,
        source_context,
        metrics,
        visual_summary,
        image_paths,
    )
    model_scores = score_code_video_indicators(
        client,
        model_name,
        simulation_description,
        indicators,
        conceptual_model,
        insights,
    )
    model_aggregate = compute_model_plausibility_from_indicator_scores(model_scores, indicators)
    model_summary = build_model_summary_object(model_scores, model_aggregate)
    face_validation = None
    if run_face_validation:
        face_validation = face_validate_code_video(
            client,
            model_name,
            simulation_description,
            indicators,
            model_summary,
        )

    score = float(model_aggregate.get("model_plausibility", 0.0))
    result = {
        "mode": "code_video",
        "status": "passed" if score >= plausibility_threshold else "failed",
        "plausibility_score": score,
        "plausibility_threshold": plausibility_threshold,
        "model": model_name,
        "indicators": indicators,
        "image_paths": [str(path) for path in image_paths],
        "visual_summary": visual_summary,
        "metrics": metrics,
        "model_insights": insights,
        "model_indicator_scores": model_scores,
        "model_aggregate": model_aggregate,
        "model_summary": model_summary,
        "face_validation": face_validation,
    }
    result["feedback_summary"] = plausibility_feedback_summary(result)
    write_json(out_dir / "plausibility_report.json", result)
    write_json(out_dir / "model_indicator_scores.json", model_scores)
    (out_dir / "source_context.txt").write_text(source_context, encoding="utf-8", errors="replace")
    return result


def load_autocred_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_autocred_state(state: dict[str, Any], path: Path) -> None:
    write_json(path, state)


def score_plausibility_with_fixed_rubric(
    csv_path: Path,
    state: dict[str, Any],
    out_dir: Path,
    model: str | None = None,
    run_column: str | None = None,
    parameter_columns: list[str] | None = None,
    plausibility_threshold: float = DEFAULT_PLAUSIBILITY_THRESHOLD,
    run_face_validation: bool = False,
) -> dict[str, Any]:
    """Score one candidate using an already frozen AutoCred state."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_openai_client()
    model_name = model or str(state.get("model") or DEFAULT_MODEL)
    selected_run_column = run_column if run_column is not None else state.get("run_column")
    selected_parameter_columns = (
        parameter_columns if parameter_columns is not None else list(state.get("parameter_columns", []))
    )
    graph_count = int(state.get("graph_count", DEFAULT_GRAPH_COUNT))
    simulation_description = str(state["simulation_description"])
    conceptual_model = dict(state["conceptual_model"])
    indicators = list(state["indicators"])
    helper_path = Path(str(state["plot_helper_path"]))
    if not helper_path.is_absolute():
        helper_path = (out_dir / helper_path).resolve()
    if not helper_path.is_file():
        raise FileNotFoundError(f"Frozen AutoCred plot helper not found: {helper_path}")

    runs = load_timeseries_csv(csv_path, selected_run_column, selected_parameter_columns)
    require_single_model_assessment(runs, csv_path)
    plot_records = generate_plots_for_runs(
        runs,
        helper_path,
        out_dir / "plots",
        graph_count=graph_count,
    )
    insights = generate_insights_for_runs(
        client,
        model_name,
        runs,
        plot_records,
        simulation_description,
        conceptual_model,
    )
    run_scores = score_run_indicators(
        client,
        model_name,
        simulation_description,
        indicators,
        list(conceptual_model.get("expected_behaviours", [])),
        insights,
    )
    model_aggregate = compute_model_plausibility_from_indicator_scores(run_scores[0], indicators)
    model_summary = build_model_summary_object(run_scores[0], model_aggregate)
    face_validation = None
    if run_face_validation:
        face_validation = face_validate(
            client,
            model_name,
            simulation_description,
            indicators,
            model_summary,
        )

    score = float(model_aggregate.get("model_plausibility", 0.0))
    result = {
        "source_csv": str(csv_path),
        "status": "passed" if score >= plausibility_threshold else "failed",
        "plausibility_score": score,
        "plausibility_threshold": plausibility_threshold,
        "model": model_name,
        "indicators": indicators,
        "plot_records": plot_records,
        "model_insights": insights[0],
        "model_indicator_scores": run_scores[0],
        "model_aggregate": model_aggregate,
        "model_summary": model_summary,
        "face_validation": face_validation,
    }
    result["feedback_summary"] = plausibility_feedback_summary(result)
    write_json(out_dir / "plausibility_report.json", result)
    write_json(out_dir / "model_indicator_scores.json", run_scores[0])
    return result


def plausibility_feedback_summary(assessment: dict[str, Any]) -> str:
    status = assessment.get("status", "unknown")
    score = float(assessment.get("plausibility_score", 0.0) or 0.0)
    threshold = float(assessment.get("plausibility_threshold", 0.0) or 0.0)
    parts = [f"AutoCred plausibility {status}: score={score:.3f}, threshold={threshold:.3f}."]

    face_validation = assessment.get("face_validation") or {}
    short_summary = face_validation.get("short_summary")
    if short_summary:
        parts.append(f"Face-validation summary: {short_summary}")

    low_scores: list[tuple[float, str, str]] = []
    model_indicator_scores = assessment.get("model_indicator_scores")
    if not model_indicator_scores:
        legacy_scores = assessment.get("run_indicator_scores") or []
        model_indicator_scores = legacy_scores[0] if legacy_scores else {}
    for item in model_indicator_scores.get("scores", []):
        try:
            item_score = float(item.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        low_scores.append(
            (
                item_score,
                str(item.get("name", "unknown indicator")),
                str(item.get("reasoning", "")),
            )
        )
    for item_score, name, reasoning in sorted(low_scores)[:3]:
        if item_score >= threshold:
            continue
        parts.append(f"Low indicator: {name}={item_score:.3f}. Reason: {reasoning}")

    insight_record = assessment.get("model_insights")
    if not insight_record:
        legacy_insights = assessment.get("insights_per_run") or []
        insight_record = legacy_insights[0] if legacy_insights else {}
    if insight_record:
        first_insight = str(insight_record.get("insights", "")).strip()
        if first_insight:
            parts.append("Assessment insight excerpt: " + first_insight[:1200])

    return "\n".join(part for part in parts if part)


def run_autocred_assessment(
    csv_path: Path,
    simulation_description: str,
    out_dir: Path,
    model: str = DEFAULT_MODEL,
    run_column: str | None = None,
    parameter_columns: list[str] | None = None,
    indicator_count: int = DEFAULT_INDICATOR_COUNT,
    graph_count: int = DEFAULT_GRAPH_COUNT,
    max_sample_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
    run_face_validation: bool = True,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = make_openai_client()

    runs = load_timeseries_csv(csv_path, run_column, parameter_columns)
    if not runs:
        raise ValueError(f"No runs found in {csv_path}")
    require_single_model_assessment(runs, csv_path)

    first_data = runs[0]["data"]
    data_columns = list(first_data.keys())
    parameter_names = sorted({name for run in runs for name in run.get("params", {}).keys()})
    conceptual_model = build_conceptual_model(
        client,
        model,
        simulation_description,
        data_columns,
        parameter_names,
    )
    conceptual_evaluation = score_conceptual_model(client, model, conceptual_model)

    helper_path = out_dir / "generated_plot_helper.py"
    example_data = sample_run_data(first_data, max_sample_rows)
    generate_plot_helper(
        client,
        model,
        simulation_description,
        conceptual_model,
        example_data,
        helper_path,
        graph_count=graph_count,
    )

    plots_dir = out_dir / "plots"
    plot_records = generate_plots_for_runs(runs, helper_path, plots_dir, graph_count=graph_count)
    indicators = generate_indicators(
        client,
        model,
        simulation_description,
        conceptual_model,
        indicator_count,
    )
    insights = generate_insights_for_runs(
        client,
        model,
        runs,
        plot_records,
        simulation_description,
        conceptual_model,
    )
    run_scores = score_run_indicators(
        client,
        model,
        simulation_description,
        indicators,
        list(conceptual_model.get("expected_behaviours", [])),
        insights,
    )
    model_aggregate = compute_model_plausibility_from_indicator_scores(run_scores[0], indicators)
    model_summary = build_model_summary_object(run_scores[0], model_aggregate)

    face_validation = None
    if run_face_validation:
        face_validation = face_validate(client, model, simulation_description, indicators, model_summary)

    result = {
        "source_csv": str(csv_path),
        "model": model,
        "conceptual_model": conceptual_model,
        "conceptual_evaluation": conceptual_evaluation,
        "indicators": indicators,
        "plot_records": plot_records,
        "model_insights": insights[0],
        "model_indicator_scores": run_scores[0],
        "model_aggregate": model_aggregate,
        "model_summary": model_summary,
        "face_validation": face_validation,
    }
    write_json(out_dir / "autocrd_result.json", result)
    write_json(out_dir / "conceptual_model.json", conceptual_model)
    write_json(out_dir / "indicators.json", {"indicators": indicators})
    write_json(out_dir / "model_indicator_scores.json", run_scores[0])
    return result


def build_batch_summary_object(
    batch_id: int,
    run_indicator_scores: list[dict[str, Any]],
    batch_aggregate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "batch_id": batch_id,
        "run_indices": [run["run_index"] for run in run_indicator_scores],
        "per_run_indicator_scores": run_indicator_scores,
        "batch_indicator_summary": batch_aggregate["per_indicator_batch"],
        "batch_credibility": batch_aggregate["batch_credibility"],
        "aggregator": batch_aggregate["aggregator"],
    }


def build_model_summary_object(
    model_indicator_scores: dict[str, Any],
    model_aggregate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "assessment_id": 1,
        "run_index": model_indicator_scores["run_index"],
        "run_id": model_indicator_scores.get("run_id"),
        "indicator_scores": model_indicator_scores,
        "model_indicator_summary": model_aggregate["per_indicator_model"],
        "model_plausibility": model_aggregate["model_plausibility"],
        "aggregator": model_aggregate["aggregator"],
    }


def require_single_model_assessment(runs: list[dict[str, Any]], csv_path: Path) -> None:
    if len(runs) != 1:
        raise ValueError(
            "AutoCred plausibility expects exactly one deterministic tuned-model "
            f"time-series assessment from {csv_path}, but found {len(runs)} groups. "
            "Remove the run grouping column from autocred.run_column or provide one CSV "
            "for the single tuned model being judged."
        )


def normalize_indicator_weights(indicators: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = copy.deepcopy(indicators)
    total = sum(float(indicator["weight"]) for indicator in normalized)
    if total <= 0.0:
        return normalized
    for indicator in normalized:
        indicator["weight"] = float(indicator["weight"]) / total
    return normalized


def summarize_run_data(data: dict[str, list[Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"columns": list(data.keys())}
    for column, values in data.items():
        numeric = [float(value) for value in values if _is_number(value)]
        if not numeric:
            continue
        summary[column] = {
            "count": len(numeric),
            "min": min(numeric),
            "max": max(numeric),
            "mean": sum(numeric) / len(numeric),
            "first": numeric[0],
            "last": numeric[-1],
        }
    return summary


def sample_run_data(data: dict[str, list[Any]], max_rows: int) -> dict[str, list[Any]]:
    if max_rows <= 0:
        return data
    lengths = [len(values) for values in data.values() if isinstance(values, list)]
    if not lengths:
        return data
    length = min(lengths)
    if length <= max_rows:
        return data
    stride = max(1, length // max_rows)
    return {key: values[::stride][:max_rows] for key, values in data.items()}


def call_text(client: OpenAI, model: str, prompt: str) -> str:
    return call_messages(client, model, [{"role": "user", "content": prompt}])


def call_messages(client: OpenAI, model: str, messages: list[dict[str, Any]]) -> str:
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def call_with_images(client: OpenAI, model: str, prompt: str, image_paths: list[Path]) -> str:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": encode_image(image_path)},
            }
        )
    return call_messages(client, model, [{"role": "user", "content": content}])


def encode_image(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _extract_fenced_block(text: str, language: str) -> str:
    pattern = rf"```{language}\s*(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    generic = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if generic:
        return generic.group(1)
    return text


def _load_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _infer_run_column(columns: Any) -> str | None:
    candidates = ["run_id", "run", "run_index", "candidate_id", "seed"]
    lower_to_original = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def _json_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _parse_csv_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run AutoCred-style plausibility assessment on one tuned-model time-series CSV.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to a time-series CSV file.")
    parser.add_argument("--description", required=True, type=Path, help="Path to a simulation description text file.")
    parser.add_argument("--out-dir", default=Path("autocrd_results"), type=Path, help="Directory for results.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--run-column", default=None, help="Optional grouping column; it must produce exactly one tuned-model assessment.")
    parser.add_argument("--parameter-columns", default="", help="Comma-separated parameter columns.")
    parser.add_argument("--indicator-count", default=4, type=int, help="Number of plausibility indicators.")
    parser.add_argument("--graph-count", default=DEFAULT_GRAPH_COUNT, type=int, help="Number of fixed plots to generate.")
    parser.add_argument("--max-sample-rows", default=DEFAULT_MAX_SAMPLE_ROWS, type=int, help="Rows sampled for plot-helper prompting.")
    parser.add_argument("--skip-face-validation", action="store_true", help="Skip final face-validation check.")
    args = parser.parse_args(argv)

    description = args.description.read_text(encoding="utf-8")
    result = run_autocred_assessment(
        csv_path=args.csv,
        simulation_description=description,
        out_dir=args.out_dir,
        model=args.model,
        run_column=args.run_column,
        parameter_columns=_parse_csv_list(args.parameter_columns),
        indicator_count=args.indicator_count,
        graph_count=args.graph_count,
        max_sample_rows=args.max_sample_rows,
        run_face_validation=not args.skip_face_validation,
    )
    print(json.dumps({"model_plausibility": result["model_aggregate"]["model_plausibility"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

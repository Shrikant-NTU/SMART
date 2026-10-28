from __future__ import annotations

import json
from typing import Any


def conceptual_model_prompt(
    simulation_description: str,
    data_columns: list[str],
    parameter_names: list[str],
) -> str:
    return f"""
You are an expert in simulation-model plausibility assessment and conceptual modeling.

You are given:
1) A high-level SIMULATION DESCRIPTION.
2) A list of DATA COLUMNS from simulation time-series output.
3) A list of PARAMETER NAMES used in the experiment.

Your task:
Construct a CONCEPTUAL MODEL for judging whether this simulation model behaves
and moves like the intended Social Force Model simulation. The model must be:
- Non-software-specific.
- Concise but precise.
- Suitable as a bridge between the intended real-world scenario, the implemented computer model, and the observed time-series behaviour.

Primary question:
"Does this tuned simulation still look and behave like a physically meaningful
Social Force Model, or did it only produce good numbers while moving in an
implausible/non-SFM way?"

Follow this JSON SCHEMA EXACTLY:

{{
  "objectives": ["..."],
  "entities": [
    {{
      "name": "...",
      "role": "agent|population|resource|environment|other",
      "state_variables": ["..."],
      "columns": ["..."]
    }}
  ],
  "state_variables": [
    {{
      "name": "...",
      "kind": "position|velocity|force|population|resource|rate|other",
      "column": "..."
    }}
  ],
  "parameters": [
    {{
      "name": "...",
      "role": "initial_condition|rate|interaction_strength|geometry|policy|other"
    }}
  ],
  "mechanisms": [
    "Plain-language description of key processes and interactions."
  ],
  "assumptions": [
    "Simplifying assumptions or uncertainties in how the world is represented."
  ],
  "exclusions": [
    "Important aspects that are purposely not modeled."
  ],
  "expected_behaviours": [
    "Qualitative expectations about plausible trajectories, mechanisms, interactions, and dynamics."
  ]
}}

Hard rules:
- Use only JSON, no extra commentary.
- Every key in the schema must be present.
- For "columns" and "column", only use names from this list:
  {json.dumps(data_columns)}
- For "parameters", only use names from this list:
  {json.dumps(parameter_names)}
- If something is not directly observable in the data, set "column": null.
- Keep descriptions short, technical, and focused on plausibility rather than generic performance.
- Do not define plausibility as high task success, low overlap, or low completion
  time alone. Those metrics are supporting evidence only.
- Explicitly include expected SFM-style movement: continuous trajectories,
  acceleration-driven motion, social repulsion, obstacle avoidance, personal
  space, and goal-directed relaxation.
- The JSON must be valid and parseable with json.loads.

SIMULATION DESCRIPTION:
{simulation_description}
"""


def conceptual_model_score_prompt(conceptual_model: dict[str, Any]) -> str:
    return f"""
You are reviewing a simulation CONCEPTUAL MODEL for plausibility assessment.

CONCEPTUAL MODEL:
{json.dumps(conceptual_model, indent=2)}

Task:
Evaluate whether the conceptual model is suitable for judging simulation-model plausibility:
1) Clarity and completeness of the intended scenario, entities, mechanisms, and assumptions.
2) Internal coherence between objectives, mechanisms, expected behaviours, and exclusions.
3) Alignment between conceptual plausibility expectations and observed data/parameters.

Output strictly this JSON:

{{
  "validity_score": 0.0,
  "strengths": ["..."],
  "weaknesses": ["..."],
  "alignment_issues": ["..."],
  "reasoning": "2-3 sentence summary explaining whether this is a good plausibility rubric foundation."
}}

Rules:
- validity_score must be between 0.0 and 1.0.
- Use 3-6 concise plausibility-focused points for strengths, weaknesses, and alignment_issues.
- No extra keys and no extra text outside the JSON.
"""


def code_video_conceptual_model_prompt(
    simulation_description: str,
    source_context: str,
    visual_summary: dict[str, Any],
) -> str:
    return f"""
You are an expert reviewer of Social Force Model simulation implementations.
Your only purpose is to judge model plausibility: whether the code and videos
show pedestrians behaving and moving like a Social Force Model. You are not
optimizing scores, and you are not judging whether the model merely wins on
aggregate metrics.

You are given:
1) A simulation/scenario description.
2) Complete tracked source files for one optimized base model.
3) A video-frame summary from that model.

Task:
Create a strict conceptual model for judging whether future candidate
implementations preserve plausible Social Force Model functioning. This is not
a performance rubric. It is an implementation-faithfulness and behavioral
plausibility rubric.

Primary question:
"Does the simulation still behave like a Social Force Model after refinement?"
If the answer is no, the candidate should be considered implausible even when
goal reach rate, overlap duration, completion time, or TOPSIS are good.

Output JSON only with exactly this schema:

{{
  "scenario_objectives": ["..."],
  "core_sfm_requirements": ["..."],
  "allowed_refinement_types": ["..."],
  "forbidden_mechanism_patterns": ["..."],
  "implementation_invariants": ["..."],
  "visual_plausibility_expectations": ["..."],
  "code_review_focus": ["..."],
  "failure_modes_to_penalize": ["..."]
}}

Strict requirements:
- Focus on preserving Social Force Model mechanics: desired-motion relaxation,
  social repulsion/contact/obstacle forces, physically meaningful acceleration,
  stable time integration, and continuous pedestrian motion.
- Treat SFM-behavior plausibility as the central criterion. Pedestrians should
  move by forces/accelerations toward goals while avoiding people and obstacles,
  not by discrete scripting, teleporting, route forcing, or metric-specific hacks.
- Explicitly forbid metric-gaming shortcuts: teleporting agents, hard-coding
  success, deleting difficult agents, suppressing collisions by hiding metrics,
  directly overriding positions without a physical mechanism, or bypassing SFM
  force/acceleration logic.
- Include expectations for supermarket task behavior: entry, service-point
  visit, checkout, exit, obstacle respect, queue/wait plausibility, and personal
  space.
- Do not reward low overlap or high goal reach unless the implementation remains
  mechanistically and physically plausible.
- Penalize any candidate whose visible movement is inconsistent with SFM even if
  its numerical optimization metrics are strong.
- Every item must be direct and checkable from complete code and video frames.
- Use JSON only. No markdown. No commentary.

SIMULATION DESCRIPTION:
{simulation_description}

VIDEO/FRAME SUMMARY:
{json.dumps(visual_summary, indent=2)}

COMPLETE TRACKED SOURCE FILES:
{source_context}
"""


def code_video_indicator_generation_prompt(
    simulation_description: str,
    conceptual_model: dict[str, Any],
    indicator_count: int = 4,
) -> str:
    return f"""
You are defining strict indicators for simulation-model plausibility from code
and video frames. The indicators must answer one central question: whether the
candidate behaves and moves like a plausible Social Force Model.

SIMULATION DESCRIPTION:
{simulation_description}

CONCEPTUAL MODEL:
{json.dumps(conceptual_model, indent=2)}

Task:
Propose exactly {indicator_count} fixed indicators that score whether a
candidate implementation remains faithful to plausible Social Force Model
functioning.

Hard requirements:
- Indicators must judge implementation/model plausibility, not performance.
- Indicators must not be proxies for TOPSIS, goal reach rate, completion time,
  or overlap metrics. Metrics can support a score only when the implementation
  and movement remain SFM-plausible.
- Indicators must be reusable for every future candidate in this same scenario.
- At least one indicator must directly evaluate SFM force/formula fidelity.
- At least one indicator must directly evaluate physically plausible motion and
  numerical integration behavior visible in frames/code.
- At least one indicator must directly evaluate absence of metric-gaming or
  non-physical shortcuts.
- At least one indicator must evaluate supermarket task/route behavior without
  rewarding hard-coded success.
- Indicators must make it possible to fail a candidate that has good numerical
  metrics but no longer behaves like an SFM.
- Weights must sum to 1.0.
- Do not make all weights equal unless clearly justified by the conceptual
  model.

Output JSON only:

{{
  "indicators": [
    {{
      "name": "...",
      "description": "One direct sentence stating exactly what is scored.",
      "weight": 0.0
    }}
  ]
}}

No extra keys. No commentary.
"""


def plotting_helper_prompt(
    simulation_description: str,
    conceptual_model: dict[str, Any],
    example_data: dict[str, list[Any]],
    graph_count: int = 3,
) -> str:
    return f"""
You are an expert data analyst evaluating whether a tuned simulation behaves
and moves like a plausible Social Force Model, using time-series data.

You are given simulation time-series data from one deterministic tuned-model assessment.
The data is provided as a Python dict mapping column names to lists of values, for example:

data = {json.dumps(example_data)[:12000]}

SIMULATION DESCRIPTION:
{simulation_description}

CONCEPTUAL MODEL:
{json.dumps(conceptual_model, indent=2)}

Your task is to write a small helper module that defines exactly one function:

    def generate_plots(data, out_dir: str) -> None:

Critical goal:
Create plots that are most useful for judging whether this tuned model produces
plausible behavior for the intended simulation scenario. The plots should expose whether
motion, interactions, constraints, and model-predicted dynamics agree with the
simulation description and conceptual model. Do not plot everything.

The plots are not for proving performance. They are for detecting whether the
motion is still SFM-like: continuous, force/acceleration driven, socially
responsive, obstacle-aware, and goal-directed.

Requirements:
- Produce exactly {graph_count} figures.
- Save them as "graph1.png", "graph2.png", ..., "graph{graph_count}.png".
- Infer the time axis from keys like "step", "t", "time", "tick", "ticks", or "iteration".
- Select only the most informative variables from the remaining keys.
- Exclude constant or nearly constant series unless clearly meaningful.
- Prefer variables that reveal plausibility: positions, trajectories, velocities,
  accelerations, goals, progress, force/acceleration components, residuals,
  nearest-neighbor distances, time-to-collision, overlaps, obstacle distances,
  active/reached states, and event labels when present.
- Include plots that can reveal implausible behaviour such as jumps, jitter,
  unexplained accelerations, wall/obstacle violation, sustained overlaps,
  unrealistic speed patterns, or motion inconsistent with goals.
- Do not choose variables only because they improve aggregate performance metrics.
  The purpose is plausibility of the simulation model and generated behaviour.
- Make the {graph_count} plots analytically distinct.
- Do not put more than 3 lines on one axes.
- Use subplots when needed.
- For fig.add_subplot(...), do not pass sharex=False or sharey=False; omit
  those arguments unless sharing with an actual axes object.
- Use clear titles and axis labels.
- Use only matplotlib and numpy.
- Do not call plt.show().
- Close each figure with plt.close().
- Create out_dir if it does not exist.
- Handle missing, irregular, and long data robustly.
- Downsample before expensive plotting when the series is long.
- If lengths do not match the time axis, truncate all series to the shortest length.
- Do not silently drop outliers; use log scale, transparent clipping, or a zoomed subplot when needed.
- Do not execute anything at import time.
- No if __name__ == "__main__".

You may assume these imports are available at module level:

import os
import numpy as np
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

Return only Python code inside a ```python code block.
"""


def indicator_generation_prompt(
    simulation_description: str,
    conceptual_model: dict[str, Any],
    indicator_count: int = 4,
) -> str:
    return f"""
You are helping define fixed evaluation criteria for simulation-model
plausibility. The criteria must judge whether the tuned simulation still
behaves and moves like a Social Force Model.

SIMULATION DESCRIPTION:
{simulation_description}

CONCEPTUAL MODEL:
{json.dumps(conceptual_model, indent=2)}

Your task is to propose exactly {indicator_count} plausibility indicators that will be used
to score each tuned model/candidate throughout the experiment.

Requirements:
- Each indicator must correspond to an important aspect of simulation plausibility given the conceptual model.
- Indicators must be reusable across different candidate models in the same scenario.
- Indicators should judge one deterministic optimized-model time-series at a time; do not assume repeated stochastic runs.
- Indicators must collectively cover different, non-overlapping aspects of plausibility.
- At least some indicators should relate to expected behaviours, mechanism fidelity, physical realism, social interaction realism, route/task consistency, or rule violations.
- Do not define indicators that only reward better aggregate optimization metrics.
  A model can have good goal/overlap/completion numbers while still being implausible.
- A candidate must be penalized if it reaches good metrics through behavior that
  is not SFM-like: scripted motion, teleportation, discontinuous jumps, direct
  route forcing, hidden collision suppression, or unphysical acceleration.
- Include task success only insofar as it supports plausibility of the intended scenario.
- Use weights to express importance.
- All weights must sum to 1.0 within a small numerical tolerance.
- Do not make all weights equal unless that is genuinely justified.

For each indicator, provide:
- "name": short string name.
- "description": one sentence explaining what plausibility property it measures.
- "weight": float between 0.0 and 1.0.

Output JSON only:

{{
  "indicators": [
    {{
      "name": "...",
      "description": "...",
      "weight": 0.0
    }}
  ]
}}

No extra keys and no extra text outside the JSON.
"""


def insight_generation_prompt(
    run_index: int,
    run_dir: str,
    simulation_description: str,
    conceptual_expectations: list[str],
    numeric_summary: dict[str, Any] | None = None,
) -> str:
    summary_text = json.dumps(numeric_summary or {}, indent=2)
    expectations_text = "\n".join(f"- {item}" for item in conceptual_expectations)
    return f"""
These are plots from tuned-model assessment {run_index + 1} ({run_dir}).
Use the simulation description, conceptual expectations, plots, and numeric summary
to judge whether the tuned simulation behaves and moves like a plausible Social
Force Model.

SIMULATION DESCRIPTION:
{simulation_description}

CONCEPTUAL EXPECTATIONS:
{expectations_text}

NUMERIC SUMMARY:
{summary_text}

Task:
Provide 4 structured, technical plausibility insights about this single tuned-model assessment.
- Describe the main patterns visible in the plots: stability, smoothness,
  goal-directedness, interactions, constraint handling, anomalies, and
  implausible dynamics.
- Explain how the observed behavior supports or violates the conceptual expectations.
- Explicitly identify any evidence that the model behaviour may be moving away
  from the intended simulation model, even if aggregate metrics look good.
- Treat non-SFM motion as a major plausibility failure even if the model has
  good goal reach, overlap, completion-time, or TOPSIS values.
- Use the numeric summary only as supporting evidence; do not ignore the plots.
- The final insight should give an overall plausibility judgment for this tuned model.

Return a numbered list of short technical paragraphs.
"""


def per_run_scoring_prompt() -> str:
    return """
You are scoring simulation-model plausibility indicators for one deterministic
tuned-model assessment. The score must reflect whether the model behaves and
moves like a plausible Social Force Model.

You will receive:
- Simulation description.
- Fixed plausibility indicators with names and weights.
- Conceptual expectations.
- Assessment insights grounded in generated plots and numeric summaries.

Important:
- Output plausibility indicator scores for this model assessment only.
- Do not compute batch-level, repeated-run, or robustness plausibility here.
- Ground every score strictly in the assessment evidence.
- Penalize behaviour that is visually or dynamically implausible even if task
  metrics are numerically good.
- Do not compensate for non-SFM motion with high task performance. A model that
  moves by shortcuts, hard-coded events, hidden metric suppression, or
  unphysical jumps should score low.
- Distinguish explicit scenario events from ordinary movement. Do not penalize
  expected spawn, wait, route-recovery, boundary-correction, or exit events
  unless they dominate normal motion or create implausible dynamics.

For each indicator, output:
- name: exact indicator name.
- weight: copy the provided weight exactly.
- score: 0.0 to 1.0 for this assessment, where 1.0 means strongly plausible and 0.0 means clearly implausible.
- reasoning: 1-3 sentences grounded in this assessment's plausibility evidence.

Output JSON only:
{
  "scores": [
    {"name": "...", "weight": 0.0, "score": 0.0, "reasoning": "..."}
  ]
}

Rules:
- Use exactly the indicators provided.
- No extra keys, no extra text.
"""


def face_validation_prompt() -> str:
    return """
You are reviewing one simulation-model plausibility assessment as an experienced
human modeler. The assessment should be strict about whether the simulation
behaves and moves like a Social Force Model.

Important: your job is to check calibration of the plausibility assessment, not to invent model improvements.
- Prefer rubric stability.
- Do not propose changes unless there is clear evidence of miscalibration in this assessment.
- Do not raise the bar simply because you can.
- Do raise an issue if high numerical performance appears to have been rewarded
  despite non-SFM code structure or non-SFM visible movement.
- Only recommend changes when specific indicators show an obviously
  incorrect or inconsistent plausibility judgment relative to the simulation
  description, conceptual model, plots, numeric summaries, and indicator
  reasoning.

You are given:
- Simulation description.
- Current plausibility indicators.
- Plausibility indicator scores with reasoning for one tuned model.
- One model-level plausibility score computed deterministically.
- Aggregator description and per-indicator summary.

Output JSON only:
{
  "short_summary": "Exactly 2 sentences summarising whether the plausibility judgment is well calibrated and the biggest issue if any.",
  "issues_found": [
    {
      "issue_id": "ISSUE-1",
      "issue": "plausibility calibration issue",
      "evidence": "specific plot, numeric, score, or reasoning evidence",
      "severity": "low|medium|high"
    }
  ],
  "recommend_change": true,
  "recommended_changes": [
    {
      "type": "indicator score adjust|weight_adjust|indicator_change",
      "issue_id": "ISSUE-1",
      "details": "Describe the minimal change needed and what it fixes."
    }
  ]
}

Hard rules:
- If no problems: set recommend_change=false, issues_found=[], recommended_changes=[].
- Every recommended_changes item must reference an existing issue_id from issues_found.
- No extra keys, no extra text.
"""


def indicator_score_adjust_prompt() -> str:
    return """
You are adjusting plausibility indicator scores for this one tuned-model assessment based on a specific face-validation issue.

You are given:
- Simulation description.
- Current plausibility indicators.
- The assessment's plausibility indicator scores with reasoning.
- The model-level aggregation summary.
- A single target issue to correct.

Goal:
- Propose the minimal set of score adjustments needed so that scores better
  match the plausibility rubric and evidence.
- Do not change indicator definitions or weights here.
- Do not invent new indicators.

Output JSON only:
{
  "adjustments": [
    {
      "run_index": 0,
      "indicator_name": "EXACT indicator name",
      "new_score": 0.0,
      "reason": "Must explicitly cite issue_id and repeat the issue text verbatim."
    }
  ]
}

Rules:
- If no adjustments are needed, output {"adjustments": []}.
- Keep changes minimal.
- No extra keys, no extra text.
"""


def indicator_update_prompt() -> str:
    return """
You are updating the plausibility indicator set and/or weights for scoring future candidate models.

Critical rule: default is no change.
- Rubric drift destroys comparability across candidates.
- Keep indicators and weights identical unless there is strong, specific evidence of miscalibration.
- Do not get stricter over time by default.
- Make the minimal change only.

Allowed changes:
1) indicator_change: clarify wording, split, merge, add, or remove indicators only if the current set cannot represent a required plausibility concept.
2) weight_adjust: adjust weights only if there is clear evidence that weights mis-rank plausibility under the current aggregator.

Constraints for weight_adjust:
- Change at most two weights.
- Each weight changes by at most 0.05.
- All weights must be in [0, 1] and sum to 1.0 exactly.

Output JSON only:
{
  "indicators": [
    {
      "name": "EXACT indicator name",
      "description": "Updated or unchanged description",
      "weight": 0.0,
      "change_reason": "If unchanged: 'no change (stability)'. If changed: MUST cite ISSUE_REF: <issue_id> | <issue text verbatim>."
    }
  ]
}

Hard rules:
- If no change is warranted, output the full list identical to current with change_reason = "no change (stability)".
- Every changed indicator must cite an issue using ISSUE_REF exactly as specified.
- No extra keys, no extra text.
"""


def code_video_insight_prompt(
    simulation_description: str,
    conceptual_model: dict[str, Any],
    source_context: str,
    metrics: dict[str, Any],
    visual_summary: dict[str, Any],
) -> str:
    return f"""
You are performing a strict plausibility review of one optimized simulation
model implementation. Your central task is to decide whether the candidate
still behaves and moves like a Social Force Model.

You are given:
- Simulation description.
- Fixed conceptual model for SFM implementation plausibility.
- Complete tracked source files for this candidate.
- Video frames attached to this request.
- Metrics and video-frame summary.

Your task:
Analyze whether this implementation preserves plausible Social Force Model
functioning. Judge code and visible behavior together. Performance metrics are
supporting context only and must not override implementation concerns.

Primary question:
"Would a knowledgeable SFM modeler recognize this as SFM-like pedestrian
motion produced by forces/accelerations, or does it look like a shortcut that
only improves metrics?"

Strict checks:
- Identify any code that bypasses SFM force/acceleration dynamics.
- Identify any direct position override, teleportation, deletion, hidden metric
  manipulation, or hard-coded success logic.
- Identify any mechanism that makes pedestrians move because of scripted route
  commands rather than SFM-style forces, acceleration, relaxation, and
  interactions.
- Check whether new mechanisms are physically meaningful and consistently wired
  through diagnostics, calibration parameters, and scenario call sites.
- Check whether forces, velocity updates, acceleration updates, and integration
  order are coherent and dimensionally plausible.
- Check whether attached frames show smooth pedestrian motion, obstacle respect,
  personal-space behavior, queue/wait plausibility, and supermarket task order.
- If evidence is insufficient for a high score, say exactly what is missing.
- If code and frames disagree, prefer the stricter interpretation and explain
  the conflict.

Return exactly 5 numbered technical findings:
1. SFM formula/mechanism fidelity.
2. Physical/numerical motion plausibility.
3. Metric-gaming or shortcut risk.
4. Scenario/task plausibility from frames.
5. Overall plausibility judgment.

Do not propose improvements. Do not discuss performance except as supporting
evidence. Do not let good TOPSIS, goal reach, overlap, or completion-time
numbers hide non-SFM behavior. Be direct and specific.

SIMULATION DESCRIPTION:
{simulation_description}

CONCEPTUAL MODEL:
{json.dumps(conceptual_model, indent=2)}

METRICS:
{json.dumps(metrics, indent=2)}

VIDEO/FRAME SUMMARY:
{json.dumps(visual_summary, indent=2)}

COMPLETE TRACKED SOURCE FILES:
{source_context}
"""


def code_video_scoring_prompt() -> str:
    return """
You are scoring one optimized simulation model for implementation plausibility.
The score must answer whether this candidate behaves and moves like a plausible
Social Force Model.

Input provided:
- Simulation description.
- Fixed implementation-plausibility indicators.
- Conceptual model.
- Code/video plausibility findings.

Scoring rules:
- Score only plausibility and SFM-faithfulness, not performance.
- A model with high goal reach or low overlap can still receive a low score if
  the code violates SFM mechanics or uses shortcuts.
- High TOPSIS, high goal reach, low overlap, or fast completion must never
  compensate for non-SFM code or visibly non-SFM movement.
- Penalize direct position hacks, hard-coded success, artificial deletion of
  agents, hidden collision suppression, force bypasses, unstable integration, or
  formulas without physical meaning.
- Penalize motion that is discontinuous, scripted, force-bypassing,
  acceleration-inconsistent, obstacle-ignoring, socially unresponsive, or
  impossible for an SFM pedestrian.
- Reward only mechanisms that remain coherent with SFM force/acceleration logic
  and visible pedestrian behavior.
- Every score must cite concrete code-level or frame-level evidence.
- If evidence is missing or ambiguous, do not assume correctness; assign a
  conservative score and say what evidence is missing.

For each indicator, output:
- name: exact indicator name.
- weight: copy the provided weight exactly.
- score: 0.0 to 1.0, where 1.0 is strongly plausible and 0.0 is clearly implausible.
- reasoning: 1-3 direct sentences citing code/frame evidence.

Output JSON only:
{
  "scores": [
    {"name": "...", "weight": 0.0, "score": 0.0, "reasoning": "..."}
  ]
}

Use exactly the provided indicators. No extra keys. No commentary.
"""


def code_video_face_validation_prompt() -> str:
    return """
You are reviewing one implementation-plausibility assessment as an experienced
Social Force Modeler. Your role is to ensure the final plausibility score is
strictly about SFM-like behavior and implementation faithfulness.

Task:
Check whether the indicator scores are calibrated to the evidence. Do not
invent model improvements. Do not reward performance metrics when the code or
frames violate SFM plausibility.

Raise a calibration issue whenever the assessment gives a high plausibility
score to a candidate whose code or frames suggest shortcut behavior,
non-physical motion, or movement that a Social Force Model should not produce.

You are given:
- Simulation description.
- Fixed implementation-plausibility indicators.
- Model-level indicator scores with reasoning.
- One deterministic model plausibility score.

Output JSON only:
{
  "short_summary": "Exactly 2 direct sentences summarizing whether the plausibility judgment is calibrated and the most important issue.",
  "issues_found": [
    {
      "issue_id": "ISSUE-1",
      "issue": "calibration issue",
      "evidence": "specific code, frame, score, or reasoning evidence",
      "severity": "low|medium|high"
    }
  ],
  "recommend_change": true,
  "recommended_changes": [
    {
      "type": "indicator score adjust|weight_adjust|indicator_change",
      "issue_id": "ISSUE-1",
      "details": "Minimal change needed and what it fixes."
    }
  ]
}

Rules:
- If no calibration problem exists, set recommend_change=false,
  issues_found=[], recommended_changes=[].
- Every recommended change must reference an existing issue_id.
- No extra keys. No commentary.
"""

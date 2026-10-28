"""Prompt templates for the SFM refinement loop.

These are intentionally editable. The agent fills the placeholders and stores
the rendered prompts in history so every iteration is reproducible.
"""

DIAGNOSIS_PROMPT_TEMPLATE = """
You are an expert in pedestrian simulation, Social Force Models, crowd
dynamics, JAX, and controlled simulation-model diagnosis.

You are being called through an API, so you have no memory beyond this prompt.
Treat the current code, outputs, metrics, and history below as the full
state of the refinement process.

The Python refinement agent is the only controller. You must not ask to run
commands or directly control the computer. You receive selected code, metrics,
output summaries, previous iteration history, and optional video frame images.
Return valid JSON only.

Overall task:
Improve both target-reaching performance and social-interaction quality through
a pool-based candidate-solution process. The current parent candidate is active
and non-dominated. It may have produced children in an earlier iteration; if it
is selected again, that means the controller is deliberately giving it another
chance to produce a new child. Diagnose it and propose child mechanism updates
for the next generation.

Scientific loop to follow:
1. Diagnose the current PNG/MP4/frame outputs and complete tracked source files.
2. Identify the main visible failures.
3. Identify likely mechanism-level causes in the code.
4. Propose exactly {max_children} child updates. Every child must contain
   exactly one mechanism change. When more than one child is requested, those
   children must be mutually different alternatives, not cumulative versions
   or combinations of one another.
5. Rank the children by priority.
6. Provide specific pseudocode for each child candidate.
7. Map every diagnosis and refinement suggestion to relevant literature.
8. Define what visual/metric evidence should be checked in the next output.

Candidate priority method:
1. Prefer the candidate that fixes the largest visible failure across multiple
   scenarios.
2. Prefer mechanism-level causes over parameter-level causes.
3. Prefer changes that are testable in the next output.
4. Prefer medium or low implementation risk when possible.
5. Do not rank first a candidate that requires rewriting the entire simulator
   unless all simpler candidates are clearly insufficient.

Previous-iteration rule:
- Do not restart from zero when history is present.
- Compare the current output to previous output/history.
- The active pool may contain candidates that have produced children before.
  Do not treat previous child production as a reason to avoid new proposals.
- If the previous candidate failed, prefer the next untried candidate from the
  previous set instead of inventing a completely new set, unless the old set is
  exhausted or invalid.
- If the previous candidate improved or partially improved the result, one new
  candidate may build on it, one may modify it, and one may target the next
  largest failure.
- If Parent candidate state contains recent_plausibility_failure_feedback,
  this call is replacing a child that failed the fixed AutoCred plausibility
  rubric. Treat that feedback as evidence. Propose a different single
  mechanism that directly addresses the plausibility failure without repeating
  the failed mechanism.

Important rules:
- One change at a time is mandatory. Each child candidate must implement one
  and only one coherent method/mechanism change.
- A mechanism change may add, modify, simplify, disable, or remove one existing
  model mechanism, but only when the evidence shows that doing so addresses a
  visible, metric, or plausibility failure. Removal or disabling counts as the
  one mechanism change for that child.
- Every proposed behavioral change must affect the next agent state through the
  existing Example 10 state-update pathway: force or decision -> acceleration
  -> velocity -> position -> state returned by single_update_active().
- A candidate may change single_update_active() directly, change an existing
  helper whose result already reaches single_update_active(), or add a helper in
  jsfm/sfm.py that is called from that pathway. A helper-only change is valid
  only when its behavioral effect reaches the returned agent state.
- The following current state-transition statements are scientist-prescribed
  and immutable. Do not propose changing, replacing, bypassing, or duplicating
  them:
    preferred = preferred_velocity(self_state)
    actual = actual_velocity(self_state)
    updated_preferred = preferred + dt * global_acceleration
    max_speed = jnp.where(self_parameters[18] > 0.0, self_parameters[18], self_parameters[2])
    updated_actual = clip_velocity(updated_preferred, max_speed)
    updated_position = self_state[:2] + dt * actual
    return updated_velocity_state(self_state, updated_position, updated_preferred, updated_actual)
- Do not propose edits to clip_velocity(), preferred_velocity(),
  actual_velocity(), or updated_velocity_state(). In particular, do not change
  the position update to use updated_actual or another new-velocity value.
- The only proposed source-code target is jsfm/sfm.py. The only additional
  editable file is calibration_parameters.yaml, and only for registration of a
  tunable scalar parameter belonging to the selected mechanism.
- Treat every other file as read-only. In particular, do not propose changes to
  example scripts, jsfm/metrics.py, jsfm/utils.py, scoring, TOPSIS, AutoCred,
  calibration execution, spawning, removal, goal detection, termination, media,
  or experiment-control code. Do not propose a new Python module.
- Do not propose metric-threshold changes, altered success definitions,
  post-update position correction, or any mechanism that improves the reported
  result without producing that improvement through the normal state update.
- When more than one child is requested, return exactly the requested number of
  genuinely different single-change alternatives. Do not put the same change
  in multiple children with only different constants, names, or minor
  implementation details. Do not make child 2 include child 1 plus another
  change.
- When exactly one child is requested, propose exactly one single mechanism
  change for that parent. Do not bundle secondary improvements into it.
- Every diagnosis, mechanism cause, and child refinement suggestion must be
  connected to at least one relevant literature source such as a paper,
  journal article, book chapter, or technical article.
- Literature mapping should be specific enough to identify the source, for
  example author/year/title/topic. If you are uncertain of the exact citation,
  mark confidence as "low" and describe the idea/source area without inventing
  fake bibliographic details.
- Every literature confidence value must include a confidence_reason explaining
  why the mapping is high, medium, or low confidence.
- Prefer Social Force Model, pedestrian dynamics, crowd simulation,
  bottleneck-flow, collision-avoidance, lane-formation, route-choice, and
  numerical-simulation literature.
- Do not use literature mapping as decoration. Explain how the source supports
  the proposed mechanism.
- Do not mix unrelated changes in one cycle.
- Do not give full Python code.
- Do not suggest superficial changes.
- Do not only tune constants such as desired speed, repulsion strength,
  pedestrian radius, relaxation time, or random seed unless clearly tied to a
  visible or metric failure and part of a deeper mechanism change.
- Prefer mechanism-level SFM improvements.
- Do not rewrite the whole code unless the selected method truly requires it;
  return pseudocode only.
- Give expected benefits and risks.
- Consider previous failed iterations.
- Avoid repeating failed ideas.
- Every candidate must be tied to visible evidence from the PNGs/MP4/frames
  or metrics.
- When multiple scenarios are evaluated, diagnose each scenario separately
  using the per-scenario metrics in scenario_metrics. Do not hide a regression
  in one scenario behind aggregate improvement in another.
- Every selected update must be testable in the next simulation output.
- Identify zero to {max_new_parameters} new scalar numeric mechanism parameters
  that should be calibrated for each child. Do not propose computed values,
  dimensions, array sizes, random seeds, booleans, or categorical choices as
  tunable parameters.
- Keep all mechanism logic inside the approved state-update pathway in
  jsfm/sfm.py. Do not require scenario-side variables, imports, hooks, or other
  file changes. If a proposal cannot be implemented within this boundary, do
  not return it as a candidate.
- Return valid JSON only, with no Markdown fences and no explanation outside
  JSON.
- Do not add background explanation, tutorial text, or Markdown.

Required JSON shape:
{{
  "parent_candidate_id": "{parent_candidate_id}",
  "current_model_judgement": "brief judgement of the current model behavior",
  "diagnosis": "main problem description",
  "per_scenario_diagnosis": [
    {{
      "scenario_name": "exampleN",
      "main_problem": "scenario-specific problem",
      "metric_evidence": "key metrics for this scenario",
      "visual_or_log_evidence": "key visible/log evidence if available"
    }}
  ],
  "main_observed_failures": ["failure 1", "failure 2"],
  "likely_mechanism_level_causes": ["cause 1", "cause 2"],
  "diagnosis_literature_mapping": [
    {{
      "diagnosis_or_cause": "diagnosis/cause text",
      "source": "author/year/title or identifiable literature source",
      "source_type": "paper | journal_article | book_chapter | technical_article | other",
      "how_it_supports_the_diagnosis": "brief connection to the observed failure",
      "confidence": "high | medium | low",
      "confidence_reason": "why this literature mapping confidence is high, medium, or low"
    }}
  ],
  "previous_candidate_assessment": {{
    "status": "none | successful | partially_successful | failed | unknown",
    "what_improved": ["..."],
    "what_became_worse": ["..."],
    "recommendation": "keep | modify | revert | not_applicable"
  }},
  "candidates": [
    {{
      "id": 1,
      "name": "candidate name",
      "target_file": "jsfm/sfm.py",
      "scenario_file_change_allowed": "none",
      "target_area": "single_update_active or a state-update helper whose output reaches it",
      "targets": "failure it targets",
      "mechanism_change": "what changes conceptually",
      "literature_basis": [
        {{
          "source": "author/year/title or identifiable literature source",
          "source_type": "paper | journal_article | book_chapter | technical_article | other",
          "relevant_concept": "concept from the source that motivates this refinement",
          "how_it_maps_to_this_code_change": "specific connection between literature and pseudocode",
          "confidence": "high | medium | low",
          "confidence_reason": "why this literature mapping confidence is high, medium, or low"
        }}
      ],
      "why_it_may_work": "why this might work",
      "why_it_may_fail": "why this might fail",
      "expected_improvement": "why this should help",
      "possible_side_effect": "what could get worse",
      "implementation_difficulty": "Low | Medium | High",
      "risk_level": "Low | Medium | High",
      "proposed_tunable_parameters": [
        {{
          "name": "descriptive_snake_case_name",
          "meaning": "physical or behavioral meaning",
          "data_type": "float | int",
          "default": 0.0,
          "minimum": 0.0,
          "maximum": 1.0,
          "mutation_size": 0.1,
          "bounds_rationale": "literature, physical constraint, or conservative fallback rationale"
        }}
      ],
      "pseudocode": "pseudocode steps",
      "priority_score": 0.0
    }},
    {{
      "id": 2,
      "name": "candidate name",
      "target_file": "jsfm/sfm.py",
      "scenario_file_change_allowed": "none",
      "target_area": "single_update_active or a state-update helper whose output reaches it",
      "targets": "failure it targets",
      "mechanism_change": "what changes conceptually",
      "literature_basis": [
        {{
          "source": "author/year/title or identifiable literature source",
          "source_type": "paper | journal_article | book_chapter | technical_article | other",
          "relevant_concept": "concept from the source that motivates this refinement",
          "how_it_maps_to_this_code_change": "specific connection between literature and pseudocode",
          "confidence": "high | medium | low",
          "confidence_reason": "why this literature mapping confidence is high, medium, or low"
        }}
      ],
      "why_it_may_work": "why this might work",
      "why_it_may_fail": "why this might fail",
      "expected_improvement": "why this should help",
      "possible_side_effect": "what could get worse",
      "implementation_difficulty": "Low | Medium | High",
      "risk_level": "Low | Medium | High",
      "proposed_tunable_parameters": [],
      "pseudocode": "pseudocode steps",
      "priority_score": 0.0
    }}
  ],
  "recommended_candidate_id": 1,
  "selection_reason": "why the highest-ranked child is first",
  "why_other_candidates_postponed": "why lower-ranked children have lower priority",
  "selected_candidate_pseudocode": {{
    "selected_candidate": "highest-ranked child candidate name",
    "purpose": "...",
    "literature_basis_summary": "short statement connecting the selected candidate to its literature basis",
    "where_it_fits": "current script/function area",
    "inputs_required": ["..."],
    "new_state_variables_required": ["..."],
    "pseudocode": "pseudocode steps",
    "expected_visual_effect": "...",
    "failure_signs_after_implementation": ["..."],
    "what_metric_or_visual_evidence_to_check_next": ["..."]
  }},
  "decision_rule_for_next_cycle": {{
    "if_improved": "build on this mechanism",
    "if_partially_improved": "modify this mechanism",
    "if_not_improved": "revert and try the next candidate"
  }}
}}

If {max_children} is greater than the two candidate objects illustrated above,
continue the same candidate-object schema until exactly {max_children}
candidates are returned. Every candidate object must include literature_basis.

Scenario/examples being evaluated:
{examples}

Parent candidate state:
{parent_state_json}

Current metrics:
{metrics_json}

Previous iteration history:
{history_json}

Output/video/frame summary:
{visual_summary}

Complete tracked source files:
{code_excerpt}
"""


CODEX_EDIT_PROMPT_TEMPLATE = """
You are Codex running inside a copied candidate workspace for a Social Force
Model experiment. The base project outside this workspace must not be touched.

The Python refinement controller selected one candidate. Apply exactly one
mechanism-level code change that implements the selected pseudocode.

Constraints:
- One change at a time is mandatory. Implement only the selected mechanism.
  Do not add a second mechanism, opportunistic improvement, unrelated cleanup,
  or alternative approach. Calibration-manifest registration for that one
  mechanism is part of the same change and is allowed under the rules below.
- The selected mechanism may be an addition, modification, simplification,
  disabling, or removal of one existing model mechanism. If disabling or
  removal is selected, remove only code inside the approved state-update scope
  that directly belongs to that mechanism. Preserve everything unrelated and
  do not remove inherited calibration-manifest entries.
- Editable-file allowlist: you may edit only jsfm/sfm.py and, conditionally,
  calibration_parameters.yaml. Treat every other existing file as read-only.
  Do not create, rename, move, or delete any file.
- In jsfm/sfm.py, behavioral edits are limited to single_update_active() and
  mechanism helpers whose outputs reach the state returned by
  single_update_active(). You may add a helper in jsfm/sfm.py only when that
  helper is called from this state-update pathway.
- Every behavioral effect of the selected mechanism must pass through the
  normal update sequence: force or decision -> acceleration -> velocity ->
  position -> state returned by single_update_active().
- A helper-only edit is allowed when single_update_active() already calls that
  helper directly or indirectly and the changed result affects its returned
  state. A textual change inside single_update_active() is not required in that
  case.
- The following current statements inside update_active_human() are
  scientist-prescribed and immutable. Keep each statement exactly equivalent;
  do not replace, bypass, duplicate, or reassign its target:
    preferred = preferred_velocity(self_state)
    actual = actual_velocity(self_state)
    updated_preferred = preferred + dt * global_acceleration
    max_speed = jnp.where(self_parameters[18] > 0.0, self_parameters[18], self_parameters[2])
    updated_actual = clip_velocity(updated_preferred, max_speed)
    updated_position = self_state[:2] + dt * actual
    return updated_velocity_state(self_state, updated_position, updated_preferred, updated_actual)
- Do not edit clip_velocity(), preferred_velocity(), actual_velocity(), or
  updated_velocity_state(). The current position equation intentionally uses
  actual, not updated_actual. It must remain unchanged.
- Do not modify step_active(), scenario scripts, jsfm/metrics.py, jsfm/utils.py,
  jsfm/__init__.py, diagnostics, waypoint logic, spawning, removal, goal
  detection, termination, scoring, TOPSIS, AutoCred, calibration execution,
  media generation, or experiment-control code.
- Do not alter metric thresholds or success definitions. Do not add post-update
  position correction, direct overlap projection, or another shortcut that
  improves reported metrics without producing the improvement through the
  normal state update.
- If the selected pseudocode cannot be implemented within this exact scope,
  make no edits and explain the scope conflict in the final message. Do not
  broaden the implementation to another file or function.
- Do not delete outputs, history, or unrelated files.
- Keep the simulation runnable with the existing run_examples.py workflow.
- Respect existing data-type and array-backend contracts. Before changing a
  helper, check whether its callers expect mutable arrays, immutable arrays,
  tensors, Python scalars, lists, dataclasses, dictionaries, or another
  project-specific representation.
- Do not partially convert a workflow from one array backend or data model to
  another. If a function starts returning a different object type or backend
  representation, convert the result back at the approved function boundary.
- For every function, method, constant, or configuration value you change,
  inspect its callers and preserve signature, return type, mutability, shape,
  units, side effects, and error behavior unless the selected mechanism
  explicitly requires changing them. Do not make a contract change that would
  require editing a caller outside the approved scope.
- The workspace contains calibration_parameters.yaml. It is the calibration
  definition owned by this LLM-generated child candidate and inherited from
  its calibrated parent.
- If the mechanism introduces a meaningful scalar numeric parameter that can
  be tuned safely, add it to calibration_parameters.yaml. Register no more
  than {max_new_parameters} new parameters.
- Total active-parameter policy: {active_parameter_policy}
- Preserve inherited manifest entries. Do not disable inherited active
  parameters only to make room for new ones.
- Change an inherited entry only if this
  mechanism intentionally changes that parameter or its source target.
- Every active manifest entry must target either:
  (1) type single_params_index with a relative Python file and non-negative
  index, or (2) type module_constant with a relative Python file and the name
  of exactly one top-level assignment whose value is a plain int/float literal.
- Any newly introduced manifest parameter must target a top-level numeric
  module constant in jsfm/sfm.py. Do not add or modify a parameter-vector index
  in jsfm/utils.py.
- For a new parameter, use the implemented source value as default. Prefer
  bounds supported by physical meaning or literature. When that is unavailable,
  use conservative fallbacks: positive coefficient 0.5x to 2x default; time
  constant 0.25x to 4x default; probability default +/- 0.2 clamped to [0, 1].
  Set mutation_size to roughly 10-15 percent of the range.
- Do not register computed expressions, denominators near zero, dimensions,
  array sizes, seeds, booleans, categorical values, or parameters whose range
  can create invalid model states. Leave calibration_parameters.yaml unchanged
  when this mechanism adds no safely tunable numeric parameter.
- Each manifest entry must contain name, calibrate, data_type, default, bounds
  with min/max, mutation_size, target, introduced_by, and rationale.
- Do not hardcode API keys or credentials.
- Do not add network calls.
- Do not execute any commands at all. Do not run python, py_compile,
  compileall, imports, tests, example scripts, run_examples.py, shell commands,
  JAX smoke checks, simulations, or media generation.
- Your role is code editing only. The Python refinement controller will run all
  validation, imports, tests, and simulations after Codex exits.
- After editing, provide a concise final message summarizing changed files.

Selected candidate JSON:
{candidate_json}

Current metrics:
{metrics_json}

Previous failed ideas to avoid:
{failed_history_json}

Complete tracked source files:
{code_excerpt}
"""


VISUAL_DIAGNOSIS_PROMPT_TEMPLATE = """
Analyze these selected frames from SFM simulation videos. Return JSON
with visible crowd-dynamics failures only. Do not propose code yet.
Do not describe normal frames unless they are needed to explain the diagnosis.

Focus on:
- stuck pedestrians
- overlaps/compression
- bottleneck behavior
- oscillation or unstable paths
- goal-reaching failure

Return:
{{
  "visual_summary": "...",
  "visible_failures": ["..."],
  "confidence": 0.0
}}
"""


JSON_REPAIR_PROMPT_TEMPLATE = """
The previous response was not valid JSON or did not match the required schema.
Return corrected valid JSON only. No Markdown fences, no commentary.
Preserve information required by the schema.

Required schema:
{schema_description}

Previous response:
{bad_response}
"""

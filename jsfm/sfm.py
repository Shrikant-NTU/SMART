import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap, lax
from functools import partial


EPS = 1e-12
SFM_DIAGNOSTIC_COLUMNS = (
    "desired_direction_x",
    "desired_direction_y",
    "driving_force_x",
    "driving_force_y",
    "social_force_x",
    "social_force_y",
    "obstacle_force_x",
    "obstacle_force_y",
    "total_force_x",
    "total_force_y",
    "sfm_predicted_acceleration_x",
    "sfm_predicted_acceleration_y",
    "nearest_obstacle_distance",
)


@jit
def clip_velocity(velocity:jnp.ndarray, max_speed:jnp.ndarray) -> jnp.ndarray:
    speed = jnp.linalg.norm(velocity)
    return lax.cond(
        speed > max_speed,
        lambda x: (x / jnp.maximum(speed, EPS)) * max_speed,
        lambda x: x,
        velocity,
    )


def preferred_velocity(state:jnp.ndarray) -> jnp.ndarray:
    return state[2:4]


def actual_velocity(state:jnp.ndarray) -> jnp.ndarray:
    if state.shape[0] >= 6:
        return state[4:6]
    return state[2:4]


def updated_velocity_state(state:jnp.ndarray, position:jnp.ndarray, preferred:jnp.ndarray, actual:jnp.ndarray) -> jnp.ndarray:
    updated_state = jnp.zeros_like(state)
    updated_state = updated_state.at[:2].set(position)
    if state.shape[0] >= 6:
        updated_state = updated_state.at[2:4].set(preferred)
        updated_state = updated_state.at[4:6].set(actual)
    else:
        updated_state = updated_state.at[2:4].set(actual)
    return updated_state


@jit
def semi_minor_axis(relative_position:jnp.ndarray, displacement:jnp.ndarray) -> jnp.ndarray:
    distance_now = jnp.linalg.norm(relative_position)
    distance_after_step = jnp.linalg.norm(relative_position - displacement)
    displacement_norm = jnp.linalg.norm(displacement)
    value = (distance_now + distance_after_step) ** 2 - displacement_norm ** 2
    return 0.5 * jnp.sqrt(jnp.maximum(value, 1e-12))


semi_minor_axis_gradient = jit(grad(semi_minor_axis, argnums=0))


@jit
def perception_weight(desired_direction:jnp.ndarray, force:jnp.ndarray, anisotropy_lambda:float) -> jnp.ndarray:
    force_norm = jnp.linalg.norm(force)
    desired_norm = jnp.linalg.norm(desired_direction)
    direction_to_source = -force / jnp.maximum(force_norm, 1e-12)
    desired_unit = desired_direction / jnp.maximum(desired_norm, 1e-12)
    cos_phi = jnp.clip(jnp.dot(desired_unit, direction_to_source), -1.0, 1.0)
    sight_half_angle_cos = jnp.cos(jnp.deg2rad(100.0))
    weight = jnp.where(cos_phi >= sight_half_angle_cos, 1.0, anisotropy_lambda)
    return jnp.where((force_norm > 1e-12) & (desired_norm > 1e-12), weight, 1.0)

@jit
def compute_edge_closest_point(reference_point:jnp.ndarray, edge:jnp.ndarray):
    @jit
    def _not_nan(reference_point:jnp.ndarray, edge:jnp.ndarray):
        a = edge[0]
        b = edge[1]
        t = (jnp.dot(reference_point - a, b - a)) / (jnp.linalg.norm(b - a) ** 2)
        t_lb = lax.cond(t>0,lambda x: x,lambda x: 0.,t)
        t_star = lax.cond(t_lb<1,lambda x: x,lambda x: 1.,t_lb)
        h = a + t_star * (b - a)
        dist = jnp.linalg.norm(h - reference_point)
        return h, dist
    closest_point, min_distance = lax.cond(
        jnp.any(jnp.isnan(edge)),
        lambda _: (jnp.array([jnp.nan, jnp.nan]), jnp.float32(1_000_000.)),
        lambda _: _not_nan(reference_point, edge),
        None)
    return closest_point, min_distance
vectorized_compute_edge_closest_point = vmap(compute_edge_closest_point, in_axes=(None, 0))

@jit
def compute_obstacle_closest_point(reference_point:jnp.ndarray, obstacle:jnp.ndarray) -> jnp.ndarray:
    @jit
    def _not_nan(reference_point:jnp.ndarray, obstacle:jnp.ndarray):
        closest_points, min_distances = vectorized_compute_edge_closest_point(reference_point, obstacle)
        return closest_points[jnp.argmin(min_distances)]
    closest_point = lax.cond(
        jnp.all(jnp.isnan(obstacle)),
        lambda _: jnp.full((2,), jnp.nan),
        lambda _: _not_nan(reference_point, obstacle),
        None)
    return closest_point 
vectorized_compute_obstacle_closest_point = vmap(compute_obstacle_closest_point, in_axes=(None, 0))

@jit
def pairwise_social_force(
    human_state:jnp.ndarray,
    other_human_state:jnp.ndarray,
    parameters:jnp.ndarray,
    other_human_parameters:jnp.ndarray,
    desired_direction:jnp.ndarray,
    same_agent:bool,
):
    @jit
    def compute_social_force(human_state:jnp.ndarray, other_human_state:jnp.ndarray, parameters:jnp.ndarray, other_human_parameters:jnp.ndarray, desired_direction:jnp.ndarray):
        diff = human_state[:2] - other_human_state[:2]
        displacement = actual_velocity(other_human_state) * other_human_parameters[16]
        b = semi_minor_axis(diff, displacement)
        grad_b = semi_minor_axis_gradient(diff, displacement)
        repulsive_acceleration = (
            parameters[4]
            / parameters[6]
            * jnp.exp(-b / parameters[6])
            * grad_b
        )
        return perception_weight(desired_direction, repulsive_acceleration, parameters[17]) * repulsive_acceleration
    pairwise_social_force = lax.cond(
        same_agent,
        lambda _: jnp.zeros((2,)),
        lambda _: compute_social_force(human_state, other_human_state, parameters, other_human_parameters, desired_direction),
        None)
    return pairwise_social_force
vectorized_pairwise_social_force = vmap(pairwise_social_force, in_axes=(None, 0, None, 0, None, 0))

@jit
def pairwise_social_force_active(
    human_state:jnp.ndarray,
    other_human_state:jnp.ndarray,
    parameters:jnp.ndarray,
    other_human_parameters:jnp.ndarray,
    desired_direction:jnp.ndarray,
    should_interact:bool,
):
    return lax.cond(
        should_interact,
        lambda _: pairwise_social_force(human_state, other_human_state, parameters, other_human_parameters, desired_direction, False),
        lambda _: jnp.zeros((2,)),
        None,
    )
vectorized_pairwise_social_force_active = vmap(pairwise_social_force_active, in_axes=(None, 0, None, 0, None, 0))

@jit
def compute_obstacle_force(human_state:jnp.ndarray, obstacle:jnp.ndarray, parameters:jnp.ndarray):
    @jit
    def _not_nan(human_state:jnp.ndarray, obstacle:jnp.ndarray, parameters:jnp.ndarray):
        diff = human_state[:2] - obstacle
        dist = jnp.linalg.norm(diff)
        niw = diff / jnp.maximum(dist, 1e-12)
        return parameters[5] / parameters[7] * jnp.exp(-dist / parameters[7]) * niw
    obstacle_acceleration = lax.cond(
        jnp.any(jnp.isnan(obstacle)),
        lambda _: jnp.zeros((2,)),
        lambda _: _not_nan(human_state, obstacle, parameters),
        None)
    return obstacle_acceleration
vectorized_compute_obstacle_force = vmap(compute_obstacle_force, in_axes=(None, 0, None))

@partial(jit, static_argnames=("dt"))
def single_update(idx:int, humans_state:jnp.ndarray, human_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:float) -> jnp.ndarray:
    self_state = humans_state[idx]
    self_parameters = parameters[idx]
    preferred = preferred_velocity(self_state)
    actual = actual_velocity(self_state)
    diff = human_goal - self_state[:2]
    dist = jnp.linalg.norm(diff)
    desired_direction = diff / jnp.maximum(dist, EPS)
    desired_acceleration = ((desired_direction * self_parameters[2]) - actual) / self_parameters[3]
    other_indices = jnp.arange(humans_state.shape[0])
    same_agent = other_indices == idx
    social_acceleration = jnp.sum(
        vectorized_pairwise_social_force(self_state, humans_state, self_parameters, parameters, desired_direction, same_agent),
        axis=0,
    )
    closest_points = vectorized_compute_obstacle_closest_point(self_state[:2], obstacles)
    num_real_obstacles = jnp.sum(~jnp.isnan(closest_points[:,0]))
    obstacle_acceleration = lax.cond(
        num_real_obstacles > 0,
        lambda _: jnp.sum(vectorized_compute_obstacle_force(self_state, closest_points, self_parameters), axis=0),
        lambda _: jnp.zeros((2,)),
        None
    )
    global_acceleration = desired_acceleration + social_acceleration + obstacle_acceleration
    updated_preferred = preferred + dt * global_acceleration
    max_speed = jnp.where(self_parameters[18] > 0.0, self_parameters[18], self_parameters[2])
    updated_actual = clip_velocity(updated_preferred, max_speed)
    updated_position = self_state[:2] + dt * actual
    return updated_velocity_state(self_state, updated_position, updated_preferred, updated_actual)
vectorized_single_update = vmap(single_update, in_axes=(0, None, 0, None, 0, None))

@partial(jit, static_argnames=("dt"))
def single_update_active(
    idx:int,
    humans_state:jnp.ndarray,
    human_goal:jnp.ndarray,
    parameters:jnp.ndarray,
    obstacles:jnp.ndarray,
    active_mask:jnp.ndarray,
    dt:float,
) -> jnp.ndarray:
    def update_active_human(_):
        self_state = humans_state[idx]
        self_parameters = parameters[idx]
        preferred = preferred_velocity(self_state)
        actual = actual_velocity(self_state)
        diff = human_goal - self_state[:2]
        dist = jnp.linalg.norm(diff)
        desired_direction = diff / jnp.maximum(dist, EPS)
        desired_acceleration = ((desired_direction * self_parameters[2]) - actual) / self_parameters[3]
        other_indices = jnp.arange(humans_state.shape[0])
        should_interact = active_mask & (other_indices != idx)
        social_acceleration = jnp.sum(
            vectorized_pairwise_social_force_active(self_state, humans_state, self_parameters, parameters, desired_direction, should_interact),
            axis=0,
        )
        closest_points = vectorized_compute_obstacle_closest_point(self_state[:2], obstacles)
        num_real_obstacles = jnp.sum(~jnp.isnan(closest_points[:, 0]))
        obstacle_acceleration = lax.cond(
            num_real_obstacles > 0,
            lambda _: jnp.sum(vectorized_compute_obstacle_force(self_state, closest_points, self_parameters), axis=0),
            lambda _: jnp.zeros((2,)),
            None,
        )
        global_acceleration = desired_acceleration + social_acceleration + obstacle_acceleration
        updated_preferred = preferred + dt * global_acceleration
        max_speed = jnp.where(self_parameters[18] > 0.0, self_parameters[18], self_parameters[2])
        updated_actual = clip_velocity(updated_preferred, max_speed)
        updated_position = self_state[:2] + dt * actual
        return updated_velocity_state(self_state, updated_position, updated_preferred, updated_actual)

    return lax.cond(active_mask[idx], update_active_human, lambda _: humans_state[idx], None)
vectorized_single_update_active = vmap(single_update_active, in_axes=(0, None, 0, None, 0, None, None))


@partial(jit, static_argnames=("dt"))
def single_sfm_diagnostics_active(
    idx:int,
    humans_state:jnp.ndarray,
    human_goal:jnp.ndarray,
    parameters:jnp.ndarray,
    obstacles:jnp.ndarray,
    active_mask:jnp.ndarray,
    dt:float,
) -> jnp.ndarray:
    def diagnostics_for_active_human(_):
        self_state = humans_state[idx]
        self_parameters = parameters[idx]
        actual = actual_velocity(self_state)
        diff = human_goal - self_state[:2]
        dist = jnp.linalg.norm(diff)
        desired_direction = diff / jnp.maximum(dist, EPS)
        driving_acceleration = ((desired_direction * self_parameters[2]) - actual) / self_parameters[3]
        other_indices = jnp.arange(humans_state.shape[0])
        should_interact = active_mask & (other_indices != idx)
        social_acceleration = jnp.sum(
            vectorized_pairwise_social_force_active(
                self_state,
                humans_state,
                self_parameters,
                parameters,
                desired_direction,
                should_interact,
            ),
            axis=0,
        )
        closest_points = vectorized_compute_obstacle_closest_point(self_state[:2], obstacles)
        valid_obstacle_points = ~jnp.isnan(closest_points[:, 0])
        num_real_obstacles = jnp.sum(valid_obstacle_points)
        obstacle_acceleration = lax.cond(
            num_real_obstacles > 0,
            lambda _: jnp.sum(vectorized_compute_obstacle_force(self_state, closest_points, self_parameters), axis=0),
            lambda _: jnp.zeros((2,)),
            None,
        )
        nearest_obstacle_distance = lax.cond(
            num_real_obstacles > 0,
            lambda _: jnp.min(
                jnp.where(
                    valid_obstacle_points,
                    jnp.linalg.norm(closest_points - self_state[:2], axis=1),
                    jnp.inf,
                )
            ),
            lambda _: jnp.nan,
            None,
        )
        total_acceleration = (
            driving_acceleration
            + social_acceleration
            + obstacle_acceleration
        )
        return jnp.concatenate(
            [
                desired_direction,
                driving_acceleration,
                social_acceleration,
                obstacle_acceleration,
                total_acceleration,
                total_acceleration,
                jnp.array([nearest_obstacle_distance]),
            ]
        )

    return lax.cond(
        active_mask[idx],
        diagnostics_for_active_human,
        lambda _: jnp.full((len(SFM_DIAGNOSTIC_COLUMNS),), jnp.nan),
        None,
    )


vectorized_single_sfm_diagnostics_active = vmap(
    single_sfm_diagnostics_active,
    in_axes=(0, None, 0, None, 0, None, None),
)


@partial(jit, static_argnames=("dt"))
def compute_sfm_diagnostics_active(
    humans_state:jnp.ndarray,
    humans_goal:jnp.ndarray,
    parameters:jnp.ndarray,
    obstacles:jnp.ndarray,
    active_mask:jnp.ndarray,
    dt:float,
) -> jnp.ndarray:
    idxs = jnp.arange(len(humans_state))
    return vectorized_single_sfm_diagnostics_active(
        idxs,
        humans_state,
        humans_goal,
        parameters,
        obstacles,
        active_mask,
        dt,
    )

@partial(jit, static_argnames=("dt"))
def step(humans_state:jnp.ndarray, humans_goal:jnp.ndarray, parameters:jnp.ndarray, obstacles:jnp.ndarray, dt:float) -> jnp.ndarray:
    idxs = jnp.arange(len(humans_state))
    updated_humans_state = vectorized_single_update(idxs, humans_state, humans_goal, parameters, obstacles, dt)
    return updated_humans_state

@partial(jit, static_argnames=("dt"))
def step_active(
    humans_state:jnp.ndarray,
    humans_goal:jnp.ndarray,
    parameters:jnp.ndarray,
    obstacles:jnp.ndarray,
    active_mask:jnp.ndarray,
    dt:float,
) -> jnp.ndarray:
    idxs = jnp.arange(len(humans_state))
    updated_humans_state = vectorized_single_update_active(idxs, humans_state, humans_goal, parameters, obstacles, active_mask, dt)
    return updated_humans_state


def update_waypoints(
    current_positions,
    routes,
    waypoint_indices,
    active_mask,
    waypoint_tolerance,
    route_lengths=None,
    waypoint_tolerances=None,
):
    pedestrian_indices = np.arange(len(routes))
    if route_lengths is None:
        last_indices = np.full(len(routes), routes.shape[1] - 1, dtype=np.int32)
    else:
        last_indices = np.asarray(route_lengths, dtype=np.int32) - 1
    waypoint_indices = np.minimum(waypoint_indices, last_indices)
    active_targets = routes[pedestrian_indices, waypoint_indices]
    if waypoint_tolerances is None:
        active_tolerances = np.full(len(routes), waypoint_tolerance, dtype=float)
    else:
        active_tolerances = waypoint_tolerances[pedestrian_indices, waypoint_indices]
    reached = np.linalg.norm(current_positions - active_targets, axis=1) <= active_tolerances
    next_indices = np.minimum(waypoint_indices + 1, last_indices)
    next_targets = routes[pedestrian_indices, next_indices]
    next_segment = next_targets - active_targets
    segment_norm = np.linalg.norm(next_segment, axis=1)
    passed = (
        (waypoint_indices < last_indices)
        & (active_tolerances > 0.0)
        & (segment_norm > 1e-9)
        & (np.sum((current_positions - active_targets) * next_segment, axis=1) > 0.0)
    )
    advance = (reached | passed) & active_mask & (waypoint_indices < last_indices)
    return np.where(
        advance,
        waypoint_indices + 1,
        waypoint_indices,
    )

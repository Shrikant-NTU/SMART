import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import os
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

DESIRED_SPEED_STD = 0.26
DESIRED_SPEED_MEAN = 1.34
MAX_SPEED_FACTOR = 1.3


def get_standard_humans_parameters(n_humans:int, seed:int=42):
    single_params = jnp.array([
        0.35415,
        75.,
        DESIRED_SPEED_MEAN,
        0.62639, #tau
        4.0, #Ai
        10,
        0.6, #Bi
        0.2,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        0.,
        2.0,
        0.5,
        1.742,
    ])
    parameters = np.array(jnp.tile(single_params, (n_humans, 1)), dtype=np.float32, copy=True)
    rng = np.random.default_rng(seed)
    desired_speeds = rng.normal(DESIRED_SPEED_MEAN, DESIRED_SPEED_STD, size=n_humans)
    desired_speeds = np.maximum(desired_speeds, 0.05)
    parameters[:, 2] = desired_speeds
    parameters[:, 18] = MAX_SPEED_FACTOR * desired_speeds
    return jnp.array(parameters)

def save_trajectory_mp4(
    all_states,
    humans_goal,
    humans_parameters,
    static_obstacles,
    output_path,
    dt,
    xlim=None,
    ylim=None,
    frame_stride=5,
    fps=20,
):
    if not FFMpegWriter.isAvailable():
        print("\nSkipping MP4: ffmpeg is not available to matplotlib.")
        print("Make sure ffmpeg.exe is on PATH, then run this example again.")
        return

    all_states = np.asarray(all_states)
    humans_goal = np.asarray(humans_goal)
    humans_parameters = np.asarray(humans_parameters)
    static_obstacles = np.asarray(static_obstacles)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    colors = list(mcolors.TABLEAU_COLORS.values())
    frames = list(range(0, len(all_states), frame_stride))
    if frames[-1] != len(all_states) - 1:
        frames.append(len(all_states) - 1)

    if xlim is None or ylim is None:
        finite_points = [all_states[:, :, :2].reshape(-1, 2), humans_goal.reshape(-1, 2)]
        if static_obstacles.size:
            obstacle_points = static_obstacles.reshape(-1, 2)
            obstacle_points = obstacle_points[~np.isnan(obstacle_points).any(axis=1)]
            if len(obstacle_points):
                finite_points.append(obstacle_points)
        points = np.concatenate(finite_points, axis=0)
        padding = 1.0
        xlim = [float(np.nanmin(points[:, 0]) - padding), float(np.nanmax(points[:, 0]) + padding)] if xlim is None else xlim
        ylim = [float(np.nanmin(points[:, 1]) - padding), float(np.nanmax(points[:, 1]) + padding)] if ylim is None else ylim

    figure, ax = plt.subplots(figsize=(10, 10))
    ax.axis("equal")
    ax.set(xlabel="X", ylabel="Y", xlim=xlim, ylim=ylim)

    for h in range(len(humans_goal)):
        ax.scatter(humans_goal[h, 0], humans_goal[h, 1], marker="*", color=colors[h % len(colors)], zorder=3)
    for obstacle in static_obstacles:
        if not np.isnan(obstacle).all():
            ax.fill(obstacle[:, :, 0], obstacle[:, :, 1], facecolor="black", edgecolor="black", zorder=4)

    paths = []
    bodies = []
    heads = []
    has_heading = all_states.shape[-1] > 4
    for h in range(all_states.shape[1]):
        color = colors[h % len(colors)]
        path, = ax.plot([], [], color=color, linewidth=1.0, zorder=1)
        body = plt.Circle(
            (all_states[0, h, 0], all_states[0, h, 1]),
            humans_parameters[h, 0],
            edgecolor=color,
            facecolor="white",
            zorder=2,
        )
        ax.add_patch(body)
        paths.append(path)
        bodies.append(body)
        if has_heading:
            head = plt.Circle((all_states[0, h, 0], all_states[0, h, 1]), 0.1, color=color, zorder=3)
            ax.add_patch(head)
            heads.append(head)

    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")

    def update(frame_index):
        k = frames[frame_index]
        for h in range(all_states.shape[1]):
            paths[h].set_data(all_states[: k + 1, h, 0], all_states[: k + 1, h, 1])
            body_center = (all_states[k, h, 0], all_states[k, h, 1])
            bodies[h].center = body_center
            if has_heading:
                theta = all_states[k, h, 4]
                radius = humans_parameters[h, 0]
                heads[h].center = (
                    body_center[0] + np.cos(theta) * radius,
                    body_center[1] + np.sin(theta) * radius,
                )
        time_text.set_text(f"t = {k * dt:.2f}s")
        return [*paths, *bodies, *heads, time_text]

    animation = FuncAnimation(figure, update, frames=len(frames), interval=1000 / fps, blit=True)
    animation.save(output_path, writer=FFMpegWriter(fps=fps), dpi=150)
    plt.close(figure)
    print(f"MP4 saved to: {output_path}")

def grid_cell_obstacle_occupancy(static_obstacles:jnp.ndarray, cell_size:float, distance_threshold:int):
    obstacle_points = static_obstacles.reshape(-1, 2)
    min_xy = jnp.floor(jnp.nanmin(obstacle_points, axis=0) / cell_size).astype(int)
    max_xy = jnp.ceil(jnp.nanmax(obstacle_points, axis=0) / cell_size).astype(int)
    grid_shape = (max_xy - min_xy) + 2 * distance_threshold
    grid = jnp.zeros((grid_shape[0], grid_shape[1], len(static_obstacles)), dtype=bool)
    for obs_idx in range(len(static_obstacles)):
        if jnp.isnan(static_obstacles[obs_idx,0,0]).any():
            continue
        obs_edges = static_obstacles[obs_idx]
        for edge in obs_edges:
            p0, p1 = edge
            if jnp.isnan(p0).any() or jnp.isnan(p1).any():
                continue
            edge_vec = p1 - p0
            edge_len = jnp.linalg.norm(edge_vec)
            n_samples = max(2, int(jnp.ceil(edge_len / (cell_size * 0.5))))
            ts = jnp.linspace(0, 1, n_samples)
            interp_points = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
            grid_obs_points = jnp.floor(interp_points / cell_size).astype(int) - min_xy + distance_threshold
            for pt in grid_obs_points:
                x, y = pt
                x_min = max(0, x - distance_threshold)
                x_max = min(grid_shape[0], x + distance_threshold + 1)
                y_min = max(0, y - distance_threshold)
                y_max = min(grid_shape[1], y + distance_threshold + 1)
                grid = grid.at[x_min:x_max, y_min:y_max, obs_idx].set(True)
    max_obstacles_per_cell = int(jnp.max(jnp.sum(grid, axis=2)))
    static_obstacles_for_each_cell = jnp.full(
        (grid_shape[0], grid_shape[1], max_obstacles_per_cell),
        jnp.nan,
    )
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            cell_occupancy = grid[i, j]
            obs_indices = jnp.where(cell_occupancy)[0]
            selected_obstacles = obs_indices
            n_selected = selected_obstacles.shape[0]
            if n_selected < max_obstacles_per_cell:
                pad_shape = (max_obstacles_per_cell - n_selected)
                pad = jnp.full(pad_shape, len(static_obstacles), dtype=static_obstacles.dtype)
                selected_obstacles = jnp.concatenate([selected_obstacles, pad], axis=0)
            static_obstacles_for_each_cell = static_obstacles_for_each_cell.at[i, j].set(selected_obstacles)
    nx, ny = grid_shape
    x_coords = jnp.arange(nx) * cell_size + (min_xy[0] - distance_threshold) * cell_size
    y_coords = jnp.arange(ny) * cell_size + (min_xy[1] - distance_threshold) * cell_size
    grid_cell_coords = jnp.stack(jnp.meshgrid(x_coords, y_coords, indexing='ij'), axis=-1)
    nan_obstacle = jnp.full((1, static_obstacles.shape[1], 2, 2), jnp.nan)
    new_static_obstacles = jnp.concatenate([static_obstacles, nan_obstacle], axis=0)
    return static_obstacles_for_each_cell, new_static_obstacles, grid_cell_coords

@partial(jit, static_argnames=('cell_size'))
def filter_obstacles(
    humans_state:jnp.ndarray, 
    static_obstacles:jnp.ndarray,
    static_obstacles_per_cell:jnp.ndarray, 
    grid_coords:jnp.ndarray, 
    cell_size:float
) -> jnp.ndarray:
    human_positions = humans_state[:, :2]
    grid_origin = grid_coords[0, 0]
    human_grid_indices = jnp.floor((human_positions - grid_origin) / cell_size).astype(int)

    @jit
    def get_human_obstacles(human_grid_index, static_obstacles_per_cell):
        x, y = human_grid_index
        valid = (x >= 0) & (y >= 0) & (x < static_obstacles_per_cell.shape[0]) & (y < static_obstacles_per_cell.shape[1])
        nan_obstacles = jnp.full((static_obstacles_per_cell.shape[2],) + static_obstacles.shape[1:], jnp.nan)
        indices = static_obstacles_per_cell[x, y].astype(jnp.int32)
        obstacles = jnp.where(valid, static_obstacles[indices], nan_obstacles)
        return obstacles

    filtered_static_obstacles = vmap(get_human_obstacles, in_axes=(0, None))(human_grid_indices, static_obstacles_per_cell)
    return filtered_static_obstacles

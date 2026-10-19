"""Grid-map construction, path planning, and particle-based robot simulation."""

from __future__ import annotations

import heapq
import io
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np


Algorithm = Literal[
    "Theta*", "A*", "D*", "RRT", "Bi-RRT", "Ant colony", "Firefly",
    "PRM", "PRM*", "APF", "Pure Pursuit",
]


def generate_sample_map(width: int, height: int) -> np.ndarray:
    """Return a binary occupancy image with openings through several walls."""
    if width < 12 or height < 12:
        raise ValueError("Map width and height must both be at least 12 pixels.")

    occupied = np.zeros((height, width), dtype=bool)
    wall_width = max(1, round(min(width, height) * 0.035))
    x1, x2 = round(width * 0.30), round(width * 0.68)
    y1, y2 = round(height * 0.32), round(height * 0.68)
    occupied[y1:y1 + wall_width, round(width * 0.08):x2] = True
    occupied[y2:y2 + wall_width, x1:round(width * 0.92)] = True
    occupied[round(height * 0.12):y2, x1:x1 + wall_width] = True
    occupied[y1:round(height * 0.90), x2:x2 + wall_width] = True

    # Doorways keep the demonstration map connected at a range of resolutions.
    occupied[y1:y1 + wall_width, round(width * 0.14):round(width * 0.23)] = False
    occupied[y2:y2 + wall_width, round(width * 0.76):round(width * 0.85)] = False
    occupied[round(height * 0.16):round(height * 0.25), x1:x1 + wall_width] = False
    occupied[round(height * 0.75):round(height * 0.84), x2:x2 + wall_width] = False
    return occupied


def image_to_occupancy(
    image_bytes: bytes,
    width: int,
    height: int,
    threshold: int = 127,
    dark_is_obstacle: bool = True,
) -> np.ndarray:
    """Resize a binary/grayscale image and convert it to an occupancy grid."""
    if not 0 <= threshold <= 255:
        raise ValueError("Image threshold must be between 0 and 255.")
    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        grayscale = image.convert("L").resize((width, height), Image.Resampling.NEAREST)
        pixels = np.asarray(grayscale, dtype=np.uint8)
    return pixels <= threshold if dark_is_obstacle else pixels >= threshold


def apply_point_table(
    occupied: np.ndarray, table_text: str
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Apply ``x,y,type`` rows, where type is obstacle, start, or goal."""
    grid = occupied.copy()
    height, width = grid.shape
    start = (1.0, 1.0)
    goal = (float(width - 2), float(height - 2))
    for line_number, raw_line in enumerate(table_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = [field.strip() for field in line.replace(";", ",").split(",")]
        if [field.lower() for field in fields] == ["x", "y", "type"]:
            continue
        if len(fields) != 3:
            raise ValueError(
                f"Point table line {line_number} must have the form x,y,type."
            )
        try:
            x, y = float(fields[0]), float(fields[1])
        except ValueError as error:
            raise ValueError(
                f"Point table line {line_number} has non-numeric coordinates."
            ) from error
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"Point table line {line_number} has invalid coordinates.")
        if not (0 <= x <= width - 1 and 0 <= y <= height - 1):
            raise ValueError(
                f"Point table line {line_number} is outside the {width}x{height} map."
            )
        cell_x, cell_y = int(round(x)), int(round(y))
        kind = fields[2].lower()
        if kind == "obstacle":
            grid[cell_y, cell_x] = True
        elif kind == "start":
            start = (x, y)
        elif kind == "goal":
            goal = (x, y)
        else:
            raise ValueError(
                f"Point table line {line_number} type must be obstacle, start, or goal."
            )

    if grid[int(round(start[1])), int(round(start[0]))]:
        raise ValueError("The selected start point lies inside an obstacle.")
    if grid[int(round(goal[1])), int(round(goal[0]))]:
        raise ValueError("The selected goal point lies inside an obstacle.")
    return grid, start, goal


_NEIGHBORS = (
    (-1, -1, math.sqrt(2)), (0, -1, 1), (1, -1, math.sqrt(2)),
    (-1, 0, 1), (1, 0, 1),
    (-1, 1, math.sqrt(2)), (0, 1, 1), (1, 1, math.sqrt(2)),
)


def _line_is_clear(grid: np.ndarray, first: tuple[int, int], second: tuple[int, int]) -> bool:
    """Check grid cells along a Bresenham line, including both endpoints."""
    x0, y0 = first
    x1, y1 = second
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx - dy
    while True:
        if grid[y0, x0]:
            return False
        if (x0, y0) == (x1, y1):
            return True
        twice_error = 2 * error
        if twice_error > -dy:
            error -= dy
            x0 += sx
        if twice_error < dx:
            error += dx
            y0 += sy


def _grid_search(
    grid: np.ndarray,
    start: tuple[float, float],
    goal: tuple[float, float],
    theta_star: bool = False,
) -> list[tuple[float, float]]:
    """Find a shortest grid route with A* or Theta*."""
    start_cell = (int(round(start[0])), int(round(start[1])))
    goal_cell = (int(round(goal[0])), int(round(goal[1])))
    height, width = grid.shape
    for label, (x, y) in (("Start", start_cell), ("Goal", goal_cell)):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{label} point is outside the map.")
        if grid[y, x]:
            raise ValueError(f"{label} point lies inside an obstacle.")
    if start_cell == goal_cell:
        return [(float(start_cell[0]), float(start_cell[1]))]
    if start_cell == goal_cell:
        return [start]

    open_set: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_set, (0.0, 0.0, start_cell))
    costs = {start_cell: 0.0}
    parents: dict[tuple[int, int], tuple[int, int]] = {start_cell: start_cell}

    while open_set:
        _, current_cost, current = heapq.heappop(open_set)
        if current_cost > costs.get(current, math.inf) + 1e-9:
            continue
        if current == goal_cell:
            route = [current]
            while route[-1] != start_cell:
                route.append(parents[route[-1]])
            route.reverse()
            if theta_star:
                route = _string_pull(grid, route)
            return [(float(x), float(y)) for x, y in route]

        for dx, dy, distance in _NEIGHBORS:
            neighbor = (current[0] + dx, current[1] + dy)
            x, y = neighbor
            if not (0 <= x < width and 0 <= y < height) or grid[y, x]:
                continue
            if dx and dy and (grid[current[1], x] or grid[y, current[0]]):
                continue
            candidate_parent = current
            candidate_cost = current_cost + distance
            if theta_star:
                ancestor = parents[current]
                if ancestor != current and _line_is_clear(grid, ancestor, neighbor):
                    shortcut = math.dist(ancestor, neighbor)
                    if costs[ancestor] + shortcut < candidate_cost:
                        candidate_parent = ancestor
                        candidate_cost = costs[ancestor] + shortcut
            if candidate_cost + 1e-9 < costs.get(neighbor, math.inf):
                costs[neighbor] = candidate_cost
                parents[neighbor] = candidate_parent
                heuristic = math.dist(neighbor, goal_cell)
                heapq.heappush(open_set, (candidate_cost + heuristic, candidate_cost, neighbor))
    raise ValueError("No collision-free path connects the selected start and goal.")


def _dstar_lite(
    grid: np.ndarray, start: tuple[float, float], goal: tuple[float, float]
) -> list[tuple[float, float]]:
    """Compute a route with D* Lite's backward g/rhs consistency search."""
    start_cell = (int(round(start[0])), int(round(start[1])))
    goal_cell = (int(round(goal[0])), int(round(goal[1])))
    height, width = grid.shape
    infinity = math.inf
    g: dict[tuple[int, int], float] = {}
    rhs: dict[tuple[int, int], float] = {goal_cell: 0.0}
    versions: dict[tuple[int, int], int] = {}
    queue: list[tuple[float, float, tuple[int, int], int]] = []

    def value(table: dict[tuple[int, int], float], node: tuple[int, int]) -> float:
        return table.get(node, infinity)

    def key(node: tuple[int, int]) -> tuple[float, float]:
        minimum = min(value(g, node), value(rhs, node))
        return minimum + math.dist(start_cell, node), minimum

    def neighbors(node: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
        result = []
        for dx, dy, distance in _NEIGHBORS:
            neighbor = (node[0] + dx, node[1] + dy)
            x, y = neighbor
            if not (0 <= x < width and 0 <= y < height) or grid[y, x]:
                continue
            if dx and dy and (grid[node[1], x] or grid[y, node[0]]):
                continue
            result.append((neighbor, distance))
        return result

    def update(node: tuple[int, int]) -> None:
        if node != goal_cell:
            rhs[node] = min(
                (cost + value(g, successor) for successor, cost in neighbors(node)),
                default=infinity,
            )
        versions[node] = versions.get(node, 0) + 1
        if value(g, node) != value(rhs, node):
            first, second = key(node)
            heapq.heappush(queue, (first, second, node, versions[node]))

    update(goal_cell)
    expansions = 0
    maximum_expansions = width * height * 20
    while queue and (
        (queue[0][0], queue[0][1]) < key(start_cell)
        or value(rhs, start_cell) != value(g, start_cell)
    ):
        first, second, node, version = heapq.heappop(queue)
        if version != versions.get(node):
            continue
        old_key = (first, second)
        new_key = key(node)
        if old_key < new_key:
            update(node)
        elif value(g, node) > value(rhs, node):
            g[node] = value(rhs, node)
            for predecessor, _ in neighbors(node):
                update(predecessor)
        else:
            g[node] = infinity
            update(node)
            for predecessor, _ in neighbors(node):
                update(predecessor)
        expansions += 1
        if expansions > maximum_expansions:
            break

    if not math.isfinite(value(g, start_cell)):
        raise ValueError("No collision-free path connects the selected start and goal.")
    route = [start_cell]
    seen = {start_cell}
    while route[-1] != goal_cell:
        current = route[-1]
        choices = [
            (cost + value(g, successor), successor)
            for successor, cost in neighbors(current)
            if successor not in seen
        ]
        if not choices:
            raise ValueError("D* could not reconstruct a collision-free path.")
        _, next_node = min(choices)
        route.append(next_node)
        seen.add(next_node)
    return [(float(x), float(y)) for x, y in _string_pull(grid, route)]


def _extend_tree(
    grid: np.ndarray,
    tree: list[tuple[int, int]],
    parents: list[int],
    sample: tuple[int, int],
    step_size: float,
) -> int | None:
    nearest_index = min(range(len(tree)), key=lambda i: math.dist(tree[i], sample))
    nearest = tree[nearest_index]
    distance = math.dist(nearest, sample)
    if distance < 1e-9:
        return None
    scale = min(step_size, distance) / distance
    candidate = (round(nearest[0] + (sample[0] - nearest[0]) * scale),
                 round(nearest[1] + (sample[1] - nearest[1]) * scale))
    if candidate == nearest or not _line_is_clear(grid, nearest, candidate):
        return None
    tree.append(candidate)
    parents.append(nearest_index)
    return len(tree) - 1


def _tree_route(tree: list[tuple[int, int]], parents: list[int], index: int) -> list[tuple[int, int]]:
    route = []
    while True:
        route.append(tree[index])
        if parents[index] == index:
            break
        index = parents[index]
    route.reverse()
    return route


def _sample_free(
    rng: np.random.Generator, grid: np.ndarray
) -> tuple[int, int]:
    height, width = grid.shape
    for _ in range(1000):
        x, y = int(rng.integers(width)), int(rng.integers(height))
        if not grid[y, x]:
            return x, y
    raise ValueError("The map has too few free cells to sample a route.")


def _rrt(
    grid: np.ndarray,
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    bidirectional: bool,
    iterations: int,
    step_size: float,
    goal_bias: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    start_node = (int(round(start[0])), int(round(start[1])))
    goal_node = (int(round(goal[0])), int(round(goal[1])))
    if _line_is_clear(grid, start_node, goal_node):
        return [(float(x), float(y)) for x, y in [start_node, goal_node]]
    first, first_parent = [start_node], [0]
    second, second_parent = [goal_node], [0]
    height, width = grid.shape
    for _ in range(iterations):
        if rng.random() < goal_bias:
            sample = goal_node
        else:
            sample = _sample_free(rng, grid)
        added = _extend_tree(grid, first, first_parent, sample, step_size)
        if added is not None:
            target = first[added]
            if not bidirectional:
                if _line_is_clear(grid, target, goal_node):
                    route = _tree_route(first, first_parent, added) + [goal_node]
                    return [(float(x), float(y)) for x, y in _string_pull(grid, route)]
            else:
                connected = None
                for _ in range(max(width, height)):
                    connected = _extend_tree(
                        grid, second, second_parent, target, max(step_size, 1.0)
                    )
                    if connected is None:
                        break
                    if second[connected] == target:
                        break
                if connected is not None and second[connected] == target:
                    left = _tree_route(first, first_parent, added)
                    right = _tree_route(second, second_parent, connected)
                    if left[0] == goal_node:
                        left, right = right, left
                    route = left + right[::-1]
                    return [(float(x), float(y)) for x, y in _string_pull(grid, route)]
        if bidirectional:
            first, second = second, first
            first_parent, second_parent = second_parent, first_parent
    raise ValueError(
        f"{'Bi-RRT' if bidirectional else 'RRT'} did not find a route; increase the planning iteration limit."
    )


def _roadmap(
    grid: np.ndarray,
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    samples: int,
    neighbors: int,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    start_node = (int(round(start[0])), int(round(start[1])))
    goal_node = (int(round(goal[0])), int(round(goal[1])))
    nodes = [start_node, goal_node]
    occupied_nodes = set(nodes)
    for _ in range(samples * 3):
        if len(nodes) >= samples + 2:
            break
        candidate = _sample_free(rng, grid)
        if candidate not in occupied_nodes:
            nodes.append(candidate)
            occupied_nodes.add(candidate)
    graph: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        nearest = sorted(
            (math.dist(node, other), other_index)
            for other_index, other in enumerate(nodes)
            if other_index != index
        )[:neighbors]
        for distance, other_index in nearest:
            if _line_is_clear(grid, node, nodes[other_index]):
                graph[index].append((other_index, distance))
                graph[other_index].append((index, distance))
    queue = [(0.0, 0)]
    costs = [math.inf] * len(nodes)
    costs[0] = 0.0
    parent = [-1] * len(nodes)
    while queue:
        _, current = heapq.heappop(queue)
        if current == 1:
            route = []
            while current >= 0:
                route.append(nodes[current])
                current = parent[current]
            route.reverse()
            return [(float(x), float(y)) for x, y in _string_pull(grid, route)]
        for neighbor, distance in graph[current]:
            candidate_cost = costs[current] + distance
            if candidate_cost < costs[neighbor]:
                costs[neighbor] = candidate_cost
                parent[neighbor] = current
                heuristic = math.dist(nodes[neighbor], goal_node)
                heapq.heappush(queue, (candidate_cost + heuristic, neighbor))
    raise ValueError(
        "The roadmap is disconnected; increase roadmap samples or connection neighbors."
    )


def _ant_colony(
    grid: np.ndarray,
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    iterations: int,
    ants: int,
    evaporation: float,
    pheromone_exponent: float,
    heuristic_exponent: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    start_node = (int(round(start[0])), int(round(start[1])))
    goal_node = (int(round(goal[0])), int(round(goal[1])))
    height, width = grid.shape
    pheromone = np.ones((height, width, 8), dtype=float)
    best_route: list[tuple[int, int]] | None = None
    best_length = math.inf
    for _ in range(iterations):
        routes: list[tuple[list[tuple[int, int]], float]] = []
        for _ in range(ants):
            current = start_node
            route = [current]
            visited = {current}
            for _ in range(width * height):
                if current == goal_node:
                    break
                possible = []
                for direction, (dx, dy, distance) in enumerate(_NEIGHBORS):
                    next_node = (current[0] + dx, current[1] + dy)
                    x, y = next_node
                    if not (0 <= x < width and 0 <= y < height) or grid[y, x] or next_node in visited:
                        continue
                    if dx and dy and (grid[current[1], x] or grid[y, current[0]]):
                        continue
                    heuristic = 1.0 / (math.dist(next_node, goal_node) + 0.5)
                    weight = (
                        pheromone[current[1], current[0], direction] ** pheromone_exponent
                        * heuristic**heuristic_exponent
                        * (1 / distance)
                    )
                    possible.append((direction, next_node, weight))
                if not possible:
                    break
                weights = np.asarray([item[2] for item in possible], dtype=float)
                weights /= weights.sum()
                selected = int(rng.choice(len(possible), p=weights))
                direction, current, _ = possible[selected]
                route.append(current)
                visited.add(current)
            if route[-1] == goal_node:
                length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
                routes.append((route, length))
                if length < best_length:
                    best_route, best_length = route, length
        pheromone *= 1.0 - evaporation
        for route, length in routes:
            deposit = 1.0 / max(length, 1.0)
            for first, second in zip(route, route[1:]):
                dx, dy = second[0] - first[0], second[1] - first[1]
                direction = next(i for i, (sx, sy, _) in enumerate(_NEIGHBORS) if (sx, sy) == (dx, dy))
                pheromone[first[1], first[0], direction] += deposit
                reverse = next(i for i, (sx, sy, _) in enumerate(_NEIGHBORS) if (sx, sy) == (-dx, -dy))
                pheromone[second[1], second[0], reverse] += deposit
    if best_route is None:
        raise ValueError("Ant colony search found no route; increase ants or iterations.")
    return [(float(x), float(y)) for x, y in _string_pull(grid, best_route)]


def _firefly(
    grid: np.ndarray,
    start: tuple[float, float],
    goal: tuple[float, float],
    *,
    population: int,
    iterations: int,
    attraction: float,
    randomness: float,
    rng: np.random.Generator,
) -> list[tuple[float, float]]:
    """Use firefly attraction to shorten and smooth collision-free waypoint routes."""
    base = _grid_search(grid, start, goal, theta_star=True)
    routes = [base.copy() for _ in range(population)]

    def cost(route: list[tuple[float, float]]) -> float:
        length = sum(math.dist(a, b) for a, b in zip(route, route[1:]))
        turns = sum(
            1.0 - float(np.dot(
                (np.asarray(route[i]) - route[i - 1]) / max(math.dist(route[i], route[i - 1]), 1e-9),
                (np.asarray(route[i + 1]) - route[i]) / max(math.dist(route[i + 1], route[i]), 1e-9),
            ))
            for i in range(1, len(route) - 1)
        )
        return length + turns * 0.2

    for firefly_index in range(population):
        route = routes[firefly_index]
        for _ in range(max(1, len(route) // 2)):
            if len(route) < 3:
                break
            first = int(rng.integers(0, len(route) - 2))
            last = int(rng.integers(first + 2, len(route)))
            cell_a = (int(round(route[first][0])), int(round(route[first][1])))
            cell_b = (int(round(route[last][0])), int(round(route[last][1])))
            if _line_is_clear(grid, cell_a, cell_b):
                route[first + 1:last] = []
    for _ in range(iterations):
        for index in range(population):
            for brighter_index in range(population):
                if cost(routes[brighter_index]) >= cost(routes[index]) or len(routes[index]) < 3:
                    continue
                route = routes[index].copy()
                point_index = int(rng.integers(1, len(route) - 1))
                reference = routes[brighter_index][
                    round(point_index * (len(routes[brighter_index]) - 1) / max(len(route) - 1, 1))
                ]
                current = route[point_index]
                candidate = (
                    round(current[0] + attraction * (reference[0] - current[0]) + rng.normal(0, randomness)),
                    round(current[1] + attraction * (reference[1] - current[1]) + rng.normal(0, randomness)),
                )
                height, width = grid.shape
                if not (0 <= candidate[0] < width and 0 <= candidate[1] < height) or grid[candidate[1], candidate[0]]:
                    continue
                previous = (int(round(route[point_index - 1][0])), int(round(route[point_index - 1][1])))
                following = (int(round(route[point_index + 1][0])), int(round(route[point_index + 1][1])))
                if _line_is_clear(grid, previous, candidate) and _line_is_clear(grid, candidate, following):
                    route[point_index] = (float(candidate[0]), float(candidate[1]))
                    if cost(route) < cost(routes[index]):
                        routes[index] = route
    best = min(routes, key=cost)
    return [(float(x), float(y)) for x, y in best]


def find_path(
    grid: np.ndarray,
    start: tuple[float, float],
    goal: tuple[float, float],
    algorithm: Algorithm,
    *,
    options: dict[str, int | float] | None = None,
    seed: int = 7,
) -> list[tuple[float, float]]:
    """Plan a collision-free route with the selected search or sampling algorithm."""
    options = options or {}
    rng = np.random.default_rng(seed)
    start_cell = (int(round(start[0])), int(round(start[1])))
    goal_cell = (int(round(goal[0])), int(round(goal[1])))
    height, width = grid.shape
    for label, (x, y) in (("Start", start_cell), ("Goal", goal_cell)):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"{label} point is outside the map.")
        if grid[y, x]:
            raise ValueError(f"{label} point lies inside an obstacle.")
    if algorithm in {"A*", "APF", "Pure Pursuit"}:
        return _grid_search(grid, start, goal)
    if algorithm == "Theta*":
        return _grid_search(grid, start, goal, theta_star=True)
    if algorithm == "D*":
        return _dstar_lite(grid, start, goal)
    if algorithm in {"RRT", "Bi-RRT"}:
        return _rrt(
            grid, start, goal, bidirectional=algorithm == "Bi-RRT",
            iterations=int(options.get("planning_iterations", 2500)),
            step_size=float(options.get("rrt_step_size", 3.0)),
            goal_bias=float(options.get("goal_bias", 0.12)), rng=rng,
        )
    if algorithm in {"PRM", "PRM*"}:
        sample_count = int(options.get("roadmap_samples", 180))
        neighbor_count = (
            max(
                2,
                int(math.ceil(
                    float(options.get("prm_star_gamma", 2.0))
                    * math.log(sample_count + 2)
                )),
            )
            if algorithm == "PRM*"
            else int(options.get("roadmap_neighbors", 12))
        )
        return _roadmap(
            grid, start, goal, samples=sample_count,
            neighbors=neighbor_count, rng=rng,
        )
    if algorithm == "Ant colony":
        return _ant_colony(
            grid, start, goal,
            iterations=int(options.get("aco_iterations", 35)),
            ants=int(options.get("aco_ants", 28)),
            evaporation=float(options.get("aco_evaporation", 0.25)),
            pheromone_exponent=float(options.get("aco_pheromone_exponent", 1.2)),
            heuristic_exponent=float(options.get("aco_heuristic_exponent", 3.0)),
            rng=rng,
        )
    if algorithm == "Firefly":
        return _firefly(
            grid, start, goal,
            population=int(options.get("firefly_population", 18)),
            iterations=int(options.get("firefly_iterations", 30)),
            attraction=float(options.get("firefly_attraction", 0.35)),
            randomness=float(options.get("firefly_randomness", 0.6)),
            rng=rng,
        )
    raise ValueError(f"Unknown navigation algorithm: {algorithm}")


def _string_pull(
    grid: np.ndarray, route: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    if len(route) < 3:
        return route
    result = [route[0]]
    anchor = 0
    while anchor < len(route) - 1:
        next_index = anchor + 1
        for candidate in range(len(route) - 1, anchor, -1):
            if _line_is_clear(grid, route[anchor], route[candidate]):
                next_index = candidate
                break
        result.append(route[next_index])
        anchor = next_index
    return result


def _unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-9 else np.zeros(2, dtype=float)


@dataclass
class Simulation:
    """A population of noisy point-robot rollouts sharing one navigation policy."""

    grid: np.ndarray
    start: tuple[float, float]
    goal: tuple[float, float]
    algorithm: Algorithm
    particles: int = 80
    speed: float = 1.4
    timestep: float = 0.35
    noise: float = 0.18
    lookahead: float = 5.0
    lidar_range: float = 7.0
    attraction: float = 1.0
    repulsion: float = 2.5
    goal_tolerance: float = 2.0
    seed: int = 7
    planner_options: dict[str, int | float] | None = None

    def __post_init__(self) -> None:
        if self.particles < 1:
            raise ValueError("Particle count must be at least 1.")
        if self.speed <= 0 or self.timestep <= 0:
            raise ValueError("Speed and timestep must be greater than zero.")
        self.rng = np.random.default_rng(self.seed)
        self.path = find_path(
            self.grid, self.start, self.goal,
            self.algorithm,
            options=self.planner_options,
            seed=self.seed,
        )
        self.positions = np.tile(np.asarray(self.start, dtype=float), (self.particles, 1))
        self.positions += self.rng.normal(0.0, 0.22, self.positions.shape)
        self.positions = self._valid_positions(self.positions, np.tile(self.start, (self.particles, 1)))
        initial_direction = np.subtract(self.goal, self.start)
        self.headings = np.full(
            self.particles, math.atan2(initial_direction[1], initial_direction[0])
        )
        self.reached = np.zeros(self.particles, dtype=bool)
        self.steps = 0
        self.mean_progress = float(np.linalg.norm(np.subtract(self.goal, self.start)))

    def _valid_positions(self, positions: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        result = positions.copy()
        height, width = self.grid.shape
        for index, point in enumerate(result):
            x = int(round(point[0]))
            y = int(round(point[1]))
            if not (0 <= x < width and 0 <= y < height) or self.grid[y, x]:
                result[index] = fallback[index]
        return result

    def _direction_to_path(self, position: np.ndarray, target_distance: float) -> np.ndarray:
        nearest = min(
            range(len(self.path)),
            key=lambda index: math.dist(position, self.path[index]),
        )
        accumulated = 0.0
        for index in range(nearest, len(self.path) - 1):
            segment = math.dist(self.path[index], self.path[index + 1])
            accumulated += segment
            if accumulated >= target_distance:
                return _unit(np.asarray(self.path[index + 1]) - position)
        return _unit(np.asarray(self.goal) - position)

    def _repulsive_force(self, position: np.ndarray) -> np.ndarray:
        radius = max(1, int(math.ceil(self.lidar_range)))
        height, width = self.grid.shape
        x, y = int(round(position[0])), int(round(position[1]))
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        obstacle_y, obstacle_x = np.nonzero(self.grid[y0:y1, x0:x1])
        if len(obstacle_x) == 0:
            return np.zeros(2, dtype=float)
        offsets = np.column_stack((obstacle_x + x0, obstacle_y + y0)) - position
        distances = np.linalg.norm(offsets, axis=1)
        visible = (distances > 1e-9) & (distances <= self.lidar_range)
        if not np.any(visible):
            return np.zeros(2, dtype=float)
        offsets, distances = offsets[visible], distances[visible]
        weights = (1.0 / distances - 1.0 / self.lidar_range) / distances**2
        return -np.sum(offsets * weights[:, None], axis=0)

    def _steering_direction(self, index: int) -> np.ndarray:
        position = self.positions[index]
        if self.algorithm == "APF":
            goal_force = self.attraction * _unit(np.asarray(self.goal) - position)
            obstacle_force = self.repulsion * self._repulsive_force(position)
            route_force = 0.30 * self._direction_to_path(position, self.lookahead)
            return _unit(goal_force + obstacle_force + route_force)
        return self._direction_to_path(
            position,
            self.lookahead if self.algorithm == "Pure Pursuit" else 1.5,
        )

    def step(self) -> None:
        """Advance every particle by one simulation tick."""
        height, width = self.grid.shape
        displacement_limit = self.speed * self.timestep
        for index, position in enumerate(self.positions.copy()):
            if self.reached[index]:
                continue
            direction = self._steering_direction(index)
            direction += self.rng.normal(0.0, self.noise, 2)
            if self.algorithm == "Pure Pursuit":
                desired_heading = math.atan2(direction[1], direction[0])
                heading_error = (
                    desired_heading - self.headings[index] + math.pi
                ) % (2 * math.pi) - math.pi
                yaw_rate = np.clip(
                    2 * self.speed * math.sin(heading_error) / max(self.lookahead, 0.1),
                    -2.0,
                    2.0,
                )
                self.headings[index] += float(yaw_rate) * self.timestep
                direction = np.asarray(
                    [math.cos(self.headings[index]), math.sin(self.headings[index])]
                )
            else:
                direction = _unit(direction)
                self.headings[index] = math.atan2(direction[1], direction[0])
            candidate = position + direction * displacement_limit
            x, y = int(round(candidate[0])), int(round(candidate[1]))
            if not (0 <= x < width and 0 <= y < height) or self.grid[y, x]:
                # Try a noise-free steering correction before keeping the robot in place.
                direction = self._steering_direction(index)
                direction = _unit(direction)
                candidate = position + direction * displacement_limit
                x, y = int(round(candidate[0])), int(round(candidate[1]))
                if not (0 <= x < width and 0 <= y < height) or self.grid[y, x]:
                    candidate = position
            self.positions[index] = candidate
            if math.dist(candidate, self.goal) <= self.goal_tolerance:
                self.reached[index] = True
        self.steps += 1
        self.mean_progress = float(
            np.mean(np.linalg.norm(self.positions - np.asarray(self.goal), axis=1))
        )

    @property
    def completion(self) -> float:
        return float(np.mean(self.reached))

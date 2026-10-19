import unittest
from io import BytesIO

import numpy as np
from PIL import Image

from navigation import (
    Simulation,
    apply_point_table,
    find_path,
    generate_sample_map,
    image_to_occupancy,
)


def _is_line_clear_for_test(
    grid: np.ndarray, first: tuple[int, int], second: tuple[int, int]
) -> bool:
    x_values = np.linspace(first[0], second[0], num=max(abs(second[0] - first[0]), abs(second[1] - first[1]), 1) * 4 + 1)
    y_values = np.linspace(first[1], second[1], num=len(x_values))
    return all(
        not grid[int(round(y)), int(round(x))]
        for x, y in zip(x_values, y_values)
    )


class MapTests(unittest.TestCase):
    def test_sample_map_has_requested_dimensions_and_clear_endpoints(self) -> None:
        grid = generate_sample_map(100, 70)
        self.assertEqual(grid.shape, (70, 100))
        self.assertFalse(grid[3, 3])
        self.assertFalse(grid[66, 96])

    def test_point_table_adds_obstacles_and_sets_endpoints(self) -> None:
        grid, start, goal = apply_point_table(
            np.zeros((12, 18), dtype=bool),
            "x,y,type\n2,3,start\n15,9,goal\n7,6,obstacle",
        )
        self.assertEqual(start, (2.0, 3.0))
        self.assertEqual(goal, (15.0, 9.0))
        self.assertTrue(grid[6, 7])

    def test_point_table_rejects_out_of_bounds_points(self) -> None:
        for point in ("11,2,obstacle", "-0.4,2,obstacle", "9.6,2,obstacle"):
            with self.subTest(point=point):
                with self.assertRaisesRegex(ValueError, "outside"):
                    apply_point_table(np.zeros((10, 10), dtype=bool), point)

    def test_binary_image_resizes_and_respects_obstacle_polarity(self) -> None:
        source = Image.fromarray(np.array([[0, 255], [255, 0]], dtype=np.uint8))
        encoded = BytesIO()
        source.save(encoded, format="PNG")
        dark_obstacles = image_to_occupancy(encoded.getvalue(), 4, 4)
        bright_obstacles = image_to_occupancy(
            encoded.getvalue(), 4, 4, dark_is_obstacle=False
        )
        self.assertEqual(dark_obstacles.shape, (4, 4))
        self.assertTrue(dark_obstacles[0, 0])
        self.assertFalse(dark_obstacles[0, 3])
        self.assertFalse(bright_obstacles[0, 0])
        self.assertTrue(bright_obstacles[0, 3])

    def test_thetastar_path_is_collision_free(self) -> None:
        grid = np.zeros((20, 24), dtype=bool)
        grid[4:17, 11] = True
        grid[10, 11] = False
        path = find_path(grid, (2, 2), (21, 17), "Theta*")
        self.assertEqual(path[0], (2.0, 2.0))
        self.assertEqual(path[-1], (21.0, 17.0))
        for first, second in zip(path, path[1:]):
            self.assertTrue(
                all(
                    not grid[int(round(y)), int(round(x))]
                    for x, y in np.linspace(first, second, num=20)
                )
            )

    def test_planner_reports_disconnected_goal(self) -> None:
        grid = np.zeros((12, 12), dtype=bool)
        grid[:, 6] = True
        with self.assertRaisesRegex(ValueError, "No collision-free path"):
            find_path(grid, (2, 2), (9, 9), "A*")

    def test_all_new_planners_return_valid_routes(self) -> None:
        grid = np.zeros((28, 36), dtype=bool)
        grid[3:24, 17] = True
        grid[13:16, 17] = False
        options = {
            "planning_iterations": 4000,
            "roadmap_samples": 220,
            "roadmap_neighbors": 16,
            "aco_iterations": 20,
            "aco_ants": 20,
            "firefly_population": 8,
            "firefly_iterations": 8,
        }
        for algorithm in ("D*", "Bi-RRT", "RRT", "Ant colony", "Firefly", "PRM", "PRM*"):
            with self.subTest(algorithm=algorithm):
                path = find_path(
                    grid, (2, 2), (33, 25), algorithm,
                    options=options, seed=12,
                )
                self.assertEqual(path[0], (2.0, 2.0))
                self.assertEqual(path[-1], (33.0, 25.0))
                for first, second in zip(path, path[1:]):
                    start_cell = (int(round(first[0])), int(round(first[1])))
                    end_cell = (int(round(second[0])), int(round(second[1])))
                    self.assertTrue(
                        _is_line_clear_for_test(grid, start_cell, end_cell),
                        f"{algorithm} produced a path through an obstacle",
                    )


class SimulationTests(unittest.TestCase):
    def test_particle_simulation_advances_and_is_seed_deterministic(self) -> None:
        grid = np.zeros((30, 40), dtype=bool)
        grid[8:22, 20] = True
        grid[15, 20] = False
        first = Simulation(grid, (2, 2), (36, 26), "Theta*", particles=12, seed=4)
        second = Simulation(grid, (2, 2), (36, 26), "Theta*", particles=12, seed=4)
        first.step()
        second.step()
        self.assertEqual(first.steps, 1)
        np.testing.assert_allclose(first.positions, second.positions)
        for x, y in first.positions:
            self.assertFalse(grid[int(round(y)), int(round(x))])

    def test_each_algorithm_advances_a_particle_population(self) -> None:
        grid = np.zeros((24, 32), dtype=bool)
        grid[7:18, 16] = True
        grid[12, 16] = False
        for algorithm in ("A*", "Theta*", "APF", "Pure Pursuit"):
            with self.subTest(algorithm=algorithm):
                simulation = Simulation(
                    grid, (2, 2), (29, 21), algorithm, particles=4, seed=2
                )
                simulation.step()
                self.assertEqual(simulation.steps, 1)
                self.assertEqual(simulation.positions.shape, (4, 2))


if __name__ == "__main__":
    unittest.main()

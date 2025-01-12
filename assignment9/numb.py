import math
import random
import time
from typing import List, Tuple, Dict
import os
from statistics import mean
import yaml
import numpy as np
from numba import jit, njit, prange, objmode


# Numba-optimized distance matrix calculation
@njit(parallel=True)
def calculate_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Calculate Euclidean distance matrix using Numba."""
    n = len(coords)
    distance_matrix = np.zeros((n, n), dtype=np.int32)

    for i in prange(n):
        x1, y1 = coords[i]
        for j in range(i + 1, n):
            x2, y2 = coords[j]
            # Calculate Euclidean distance
            dx = x2 - x1
            dy = y2 - y1
            dist = round(math.sqrt(dx * dx + dy * dy))
            # Matrix is symmetric
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    return distance_matrix


# Numba-optimized cost calculation
@njit
def calculate_total_cost_numba(
    path: np.ndarray, distance_matrix: np.ndarray, cost_lookup_table: np.ndarray
) -> int:
    """Calculate total cost of a path using Numba."""
    total_cost = 0

    # Sum distances between consecutive nodes
    for i in range(len(path) - 1):
        total_cost += distance_matrix[path[i], path[i + 1]]

    # Sum node costs (excluding duplicate nodes)
    used_nodes = np.zeros(len(cost_lookup_table), dtype=np.bool_)
    for i in range(len(path) - 1):  # Exclude last node as it's same as first
        used_nodes[path[i]] = True

    for i in range(len(cost_lookup_table)):
        if used_nodes[i]:
            total_cost += cost_lookup_table[i]

    return total_cost


# Numba-optimized local search
@njit
def local_search_numba_old(
    start_solution: np.ndarray,
    distance_matrix: np.ndarray,
    cost_lookup_table: np.ndarray,
    k: int,
    max_iterations: int = 100,
) -> np.ndarray:
    """Apply local search with Numba optimization."""
    n = len(distance_matrix)

    # Initialize solution
    current_solution = start_solution.copy()

    # Initialize available nodes
    available_nodes = np.ones(n, dtype=np.bool_)
    for node in current_solution:
        available_nodes[node] = False

    # Calculate initial cost
    current_cost = calculate_total_cost_numba(
        current_solution, distance_matrix, cost_lookup_table
    )

    # Local search
    iteration = 0
    while iteration < max_iterations:
        improved = False

        # Try 2-opt moves
        for i in range(1, len(current_solution) - 2):
            if improved:
                break
            for j in range(i + 1, len(current_solution) - 1):
                # Create new solution with 2-opt move
                new_solution = current_solution.copy()
                # Reverse segment
                new_solution[i : j + 1] = new_solution[i : j + 1][::-1]

                new_cost = calculate_total_cost_numba(
                    new_solution, distance_matrix, cost_lookup_table
                )

                if new_cost < current_cost:
                    current_solution = new_solution
                    current_cost = new_cost
                    improved = True
                    break

        # Try node replacements
        if not improved:
            unused_nodes = np.where(available_nodes)[0]
            for i in range(1, len(current_solution) - 1):
                if improved:
                    break
                current_node = current_solution[i]
                prev_node = current_solution[i - 1]
                next_node = current_solution[i + 1]

                for new_node in unused_nodes:
                    # Calculate cost difference
                    old_distance = (
                        distance_matrix[prev_node, current_node]
                        + distance_matrix[current_node, next_node]
                    )
                    new_distance = (
                        distance_matrix[prev_node, new_node]
                        + distance_matrix[new_node, next_node]
                    )

                    cost_diff = (
                        new_distance
                        - old_distance
                        + cost_lookup_table[new_node]
                        - cost_lookup_table[current_node]
                    )

                    if cost_diff < 0:
                        # Update solution
                        current_solution[i] = new_node
                        current_cost += cost_diff
                        available_nodes[current_node] = True
                        available_nodes[new_node] = False
                        improved = True
                        break

        if not improved:
            break
        iteration += 1

    return current_solution


def local_search_numba(
    start_solution: np.ndarray,
    distance_matrix: np.ndarray,
    cost_lookup_table: np.ndarray,
    k: int,
    max_iterations: int = 100,
) -> np.ndarray:
    pass


@njit
def evolutionary_algorithm(
    distance_matrix: np.ndarray,
    cost_lookup_table: np.ndarray,
    k: int,
    population_size: int = 20,
    iterations: int = 10,
):
    """
        Elite population
        Steady-state evolutionary algorithm
        Parents selected from the population with the uniform probability
        All elements in the population have to be unique
        Recombination: We locate in the offspring all common nodes and edges and fill the rest of the
    solution at random.

        1. Initialize the population with random solutions
        2. Select parents uniformly at random
    """

    n = len(distance_matrix)
    population = []

    # Initialize the population with random solutions
    for i in range(population_size):
        path = random_solution()
        population.append(path)

    cost = [
        calculate_total_cost_numba(x, distance_matrix, cost_lookup_table)
        for x in population
    ]

    for i in range(iterations):
        # Steady-state evolutionary algorithm
        # Select parents uniformly at random
        p = np.random.permutation(population_size)
        parent1 = population[p[0]]
        parent2 = population[p[1]]

        offspring = recombination(parent1, parent2, n, k)

        # Local search
        offspring = local_search_numba(
            offspring, distance_matrix, cost_lookup_table, k, max_iterations=10
        )

        # Calculate cost
        offspring_cost = calculate_total_cost_numba(
            offspring, distance_matrix, cost_lookup_table
        )

        # Replace worst solution
        worst_index = cost.index(max(cost))
        if offspring_cost < cost[worst_index]:
            population[worst_index] = offspring
            cost[worst_index] = offspring_cost

    # Fidn the best solution
    best_index = cost.index(min(cost))
    return population[best_index]


@njit
def recombination(
    parent1: np.ndarray, parent2: np.ndarray, n: int = 200, k: int = 100
) -> np.ndarray:
    """Recombination of two parents."""
    common_nodes = []

    # Common edges
    edges_1 = [(parent1[i], parent1[i + 1]) for i in range(len(parent1) - 1)]
    edges_2 = [(parent2[i], parent2[i + 1]) for i in range(len(parent2) - 1)]
    # common_edges = set(edges_1) + set(edges_2)
    common_edges = [x for x in edges_1 if x in edges_2]

    new_path = list()
    i = 0
    added = set()

    # Recombination of two parents - fill the rest of the solution at random
    for a, b in common_edges:
        if a not in added:
            new_path.append(a)
            added.add(a)
        if b not in added:
            new_path.append(b)
            added.add(b)
        common_nodes.append(a)
        common_nodes.append(b)

    # Fill the rest of the solution at random
    for i in np.random.permutation(n):
        if len(new_path) == k:
            break

        if i not in common_nodes:
            new_path.append(i)

    return np.array(new_path)


def read_csv(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read CSV file and return numpy arrays for coordinates and costs."""
    coords = []
    costs = []
    try:
        with open(filename, "r") as file:
            for line in file:
                x, y, cost = map(int, line.strip().split(";"))
                coords.append((x, y))
                costs.append(cost)
    except FileNotFoundError:
        print(f"Error: File {filename} not found!")
        return np.array([]), np.array([])

    return np.array(coords, dtype=np.int32), np.array(costs, dtype=np.int32)


@njit
def random_solution(n: int = 200, k: int = 100) -> np.ndarray:
    """Generate random solution."""
    path = np.zeros(n, dtype=np.int32)
    path = np.random.permutation(n)[:k]
    return path


def main(input_file: str, algorithm_type: str):
    """Main function with Numba optimization."""
    # Read data
    coords, cost_lookup_table = read_csv(os.path.join("../data", input_file))
    if len(coords) == 0:
        return

    elements = len(coords)
    k = math.ceil(elements / 2.0)

    # Calculate distance matrix
    distance_matrix = calculate_distance_matrix(coords)

    print(f"Number of nodes: {elements}")
    print("-" * 80)

    # Generate solutions
    n_solutions = 200
    solutions = []
    total_costs = []
    execution_times = []

    for i in range(n_solutions):
        start_time = time.time()

        path = random_solution(elements, k)

        if algorithm_type.startswith("local"):
            path = local_search_numba(path, distance_matrix, cost_lookup_table, k)
        elif algorithm_type.startswith("evol"):
            path = evolutionary_algorithm(
                distance_matrix, cost_lookup_table, k, iterations=100
            )
        else:
            # Random solution
            path = random_solution(elements, k)

        # Calculate cost
        total_cost = calculate_total_cost_numba(
            path, distance_matrix, cost_lookup_table
        )

        # Store results
        solutions.append(path.tolist())
        total_costs.append(total_cost)
        execution_times.append(time.time() - start_time)

    # Calculate and save statistics
    avg_time = mean(execution_times)
    print(f"Average execution time: {avg_time:.2f} s")
    dump_to_file(
        input_file, f"{algorithm_type}_numba", solutions, total_costs, avg_time
    )


def dump_to_file(
    input_file: str,
    algorithm_type: str,
    solutions: List[List[int]],
    total_costs: List[int],
    execution_time: float,
    additional_string: str = "",
) -> bool:
    """Save results to file in YAML format."""

    # Calculate statistics
    average_cost = mean(total_costs)
    worst_index = total_costs.index(max(total_costs))
    worst_cost = total_costs[worst_index]
    worst_path = solutions[worst_index]

    best_index = total_costs.index(min(total_costs))
    best_cost = total_costs[best_index]
    best_path = solutions[best_index]

    # Print results
    print(f"Average cost: {average_cost}")
    print(f"Worst cost: {worst_cost}")
    print(f"Best cost: {best_cost}")
    print(f"Execution time: {execution_time} ms")

    # Create results directory
    os.makedirs("./results", exist_ok=True)

    # Prepare data structure for YAML
    results = {
        "average_cost": float(average_cost),
        "worst_solution": {
            "cost": int(worst_cost),
            "path": [int(x) for x in worst_path],
        },
        "best_solution": {"cost": int(best_cost), "path": [int(x) for x in best_path]},
        "execution_time_ms": float(execution_time),
        "metadata": {"algorithm": algorithm_type, "input_file": input_file},
    }

    # Create filename
    additional_string = f"_{additional_string}" if additional_string else ""
    filename = f"./results/{algorithm_type}_{input_file}{additional_string}.yaml"

    # Write results in YAML format
    with open(filename, "w") as file:
        yaml.dump(results, file, default_flow_style=False, sort_keys=False)

    return True


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file> <algorithm_type>")
        sys.exit(1)

    main(sys.argv[1], sys.argv[2])

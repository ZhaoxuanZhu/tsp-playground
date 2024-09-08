import random
from typing import List, Tuple
from interface import TSPSolver
from tsp_problem import TSPProblem, generate_random_points
from solvers.concorde_solver import ConcordeSolver


class RandomSamplingSolver(TSPSolver):
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples

    def solve(self, problem: TSPProblem) -> Tuple[List[int], float, float]:
        import time

        best_tour = None
        best_length = float("inf")

        start_time = time.time()

        for _ in range(self.num_samples):
            # Generate a random tour
            tour = list(range(problem.num_points))
            random.shuffle(tour)

            # Calculate the tour length
            length = problem.calculate_tour_length(tour)

            # Update the best tour if this one is better
            if length < best_length:
                best_tour = tour
                best_length = length

        solver_time = (time.time() - start_time) * 1000

        return best_tour, best_length, solver_time


def compare_solvers(problem: TSPProblem):
    concorde_solver = ConcordeSolver()
    random_solver = RandomSamplingSolver(num_samples=10000)

    print("Solving with Concorde:")
    concorde_tour, concorde_length, concorde_time = concorde_solver.solve(problem)
    print(f"Tour length: {concorde_length}")
    print(f"Solver time: {concorde_time:.2f} ms")

    print("\nSolving with Random Sampling:")
    random_tour, random_length, random_time = random_solver.solve(problem)
    print(f"Tour length: {random_length}")
    print(f"Solver time: {random_time:.2f} ms")

    print(f"\nDifference in tour length: {random_length - concorde_length}")
    print(f"Difference in solver time: {random_time - concorde_time:.2f} ms")

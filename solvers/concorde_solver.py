from typing import List, Tuple
from concorde.tsp import TSPSolver as ConcordeTSPSolver
from solvers.interface import TSPSolver
from tsp_problem import TSPProblem


class ConcordeSolver(TSPSolver):
    """
    Solve the TSP problem using the Concorde solver.

    :param problem: TSPProblem instance
    :return: tuple containing the optimal tour and its length
    """

    def solve(self, problem: TSPProblem) -> Tuple[List[int], float, float]:
        import time

        # Create a Concorde TSP solver instance
        solver = ConcordeTSPSolver.from_data(problem.points[:, 0], problem.points[:, 1], norm="EUC_2D")

        # Record the start time
        start_time = time.time()

        # Solve the TSP with minimal output
        solution = solver.solve(verbose=False)

        # Calculate the solver time in milliseconds
        solver_time = (time.time() - start_time) * 1000

        return solution.tour, solution.optimal_value, solver_time

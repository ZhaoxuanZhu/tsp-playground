from abc import ABC, abstractmethod
from tsp_problem import TSPProblem
from typing import Tuple, List


class TSPSolver(ABC):
    @abstractmethod
    def solve(self, problem: TSPProblem) -> Tuple[List[int], float, float]:
        """
        Solve the TSP problem.

        :param problem: TSPProblem instance
        :return: tuple containing the optimal tour (as a list of indices), its length, and the solver time in milliseconds
        """
        pass

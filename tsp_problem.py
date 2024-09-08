import numpy as np
from typing import List, Optional
import torch
from dataclasses import dataclass


def generate_random_points(n, min_coord=0, max_coord=100):
    """
    Generate n random 2D points within the specified coordinate range.

    :param n: Number of points to generate
    :param min_coord: Minimum coordinate value (default 0)
    :param max_coord: Maximum coordinate value (default 100)
    :return: numpy array of shape (n, 2) containing the random points
    """
    return np.random.uniform(min_coord, max_coord, size=(n, 2))


class TSPProblem:
    """
    Represents a Traveling Salesman Problem (TSP) instance.

    This class encapsulates the data and methods necessary to define and work with a TSP,
    including the set of points (cities), and the number of points.

    Attributes:
        points (np.ndarray): A numpy array of shape (n, 2) representing the coordinates of the cities.
        num_points (int): The number of cities in the TSP.

    Methods:
        calculate_tour_length(tour): Calculates the length of a given tour.
    """

    def __init__(self, points: np.ndarray, solution: List[int] = None):
        self.points = points
        self.num_points = len(self.points)
        self._solution = solution
        self._solution_tour_length = self.calculate_tour_length(solution) if solution is not None else None

    def calculate_tour_length(self, tour: List[int]) -> float:
        length = 0
        for i in range(len(tour)):
            length += np.linalg.norm(self.points[tour[i]] - self.points[tour[(i + 1) % len(tour)]])
        return length

    def set_solution(self, solution: List[int]):
        self._solution = solution
        self._solution_tour_length = self.calculate_tour_length(solution)

    @property
    def solution(self):
        return self._solution

    @property
    def solution_tour_length(self):
        return self._solution_tour_length


@dataclass
class TSPBatch:
    points: torch.Tensor
    padding_mask: torch.Tensor
    solutions: Optional[torch.Tensor]
    solution_tour_lengths: Optional[torch.Tensor]

    def to(self, device):
        return TSPBatch(
            self.points.to(device),
            self.padding_mask.to(device),
            self.solutions.to(device),
            self.solution_tour_lengths.to(device),
        )

    @staticmethod
    def collate_fn(
        batch: List["TSPProblem"],
    ) -> "TSPBatch":
        """
        Custom collate function for batching TSP problems using PyTorch.

        :param batch: A list of TSPProblem instances
        :return: A TSPBatch containing batched points, padding mask, and solutions as PyTorch tensors
        """
        points = [torch.tensor(problem.points, dtype=torch.float32) for problem in batch]
        solutions = [
            (torch.tensor(problem.solution, dtype=torch.long) if problem.solution is not None else None)
            for problem in batch
        ]

        # Pad points to the same size
        max_points = max(problem.num_points for problem in batch)

        padded_points = [
            torch.nn.functional.pad(p, (0, 0, 0, max_points - p.shape[0]), mode="constant") for p in points
        ]

        # Create padding mask
        padding_mask = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(problem.num_points),
                        torch.zeros(max_points - problem.num_points),
                    ]
                )
                for problem in batch
            ]
        ).bool()

        # Pad solutions to the same size
        max_points += 1
        padded_solutions = [
            (
                torch.nn.functional.pad(s, (0, max_points - s.shape[0]), mode="constant", value=-1)
                if s is not None
                else torch.full((max_points,), -1, dtype=torch.long)
            )
            for s in solutions
        ]

        # Stack tensors
        points_tensor = torch.stack(padded_points)
        solutions_tensor = torch.stack(padded_solutions)
        solution_tour_lengths = torch.stack(
            [
                (
                    torch.tensor(problem.solution_tour_length, dtype=torch.float32)
                    if problem.solution is not None
                    else torch.tensor(0, dtype=torch.float32)
                )
                for problem in batch
            ]
        )

        return TSPBatch(points_tensor, padding_mask, solutions_tensor, solution_tour_lengths)

    def __getitem__(self, index: int) -> TSPProblem:
        """
        Retrieve a TSPProblem instance at the specified index.

        Args:
            index (int): The index of the TSPProblem to retrieve.

        Returns:
            TSPProblem: The TSPProblem instance at the specified index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < 0 or index >= len(self.points):
            raise IndexError("Index out of range")

        return TSPProblem(
            points=self.points[index][: self.padding_mask[index].sum().int()],
            solution=self.solutions[index][self.solutions[index] != -1] if self.solutions is not None else None,
        )

    def __iter__(self):
        for i in range(len(self.points)):
            yield self[i]

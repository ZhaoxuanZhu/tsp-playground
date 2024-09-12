from torch.utils.data import Dataset, DataLoader
from typing import Optional
import numpy as np
from tsp_problem import TSPProblem, TSPBatch
from solvers.concorde_solver import ConcordeSolver
import os
from dataclasses import dataclass
import pickle


def generate_random_points(n, min_coord=-1, max_coord=1):
    """
    Generate n random 2D points within the specified coordinate range.

    :param n: Number of points to generate
    :param min_coord: Minimum coordinate value (default -1)
    :param max_coord: Maximum coordinate value (default 1)
    :return: numpy array of shape (n, 2) containing the random points
    """
    return np.random.uniform(min_coord, max_coord, size=(n, 2))


@dataclass
class DatasetGenerationParams:
    num_samples: int = 100
    min_points: int = 5
    max_points: int = 20
    seed: int = None
    save_path: str = None


class TSPDataset(Dataset):
    def __init__(self, load_path: Optional[str] = None, dataset_params: Optional[DatasetGenerationParams] = None):
        if load_path is None and dataset_params is None:
            raise ValueError("Either load_path or dataset_params must be provided")
        if load_path is not None and os.path.exists(load_path):
            self.problems = self._load_problems(load_path)
            self.num_samples = len(self.problems)
        else:
            self.problems = self._generate_problems(dataset_params)
            self.num_samples = dataset_params.num_samples

    def _generate_problems(self, dataset_params: DatasetGenerationParams):
        if dataset_params.seed is not None:
            np.random.seed(dataset_params.seed)
        problems = []
        for _ in range(dataset_params.num_samples):
            num_points = np.random.randint(dataset_params.min_points, dataset_params.max_points + 1)
            points = generate_random_points(num_points)
            problem = TSPProblem(points)
            solution, _, _ = ConcordeSolver().solve(problem)
            solution = solution.tolist() + [solution[0]]  # Return to the depot
            problem.set_solution(solution)
            problems.append(problem)

        if dataset_params.save_path:
            self._save_to_pickle(problems, dataset_params.save_path)

        return problems

    def _save_to_pickle(self, problems, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            pickle.dump(problems, f)

    def _load_problems(self, load_path: str):
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"File {load_path} not found")
        with open(load_path, "rb") as f:
            problems = pickle.load(f)
        return problems

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.problems[idx]


def get_tsp_dataloader(batch_size=32, num_samples=100, min_points=5, max_points=20, seed=None, load_path=None):
    dataset = TSPDataset(
        load_path, DatasetGenerationParams(num_samples, min_points, max_points, seed, save_path=load_path)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=TSPBatch.collate_fn)

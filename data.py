from torch.utils.data import Dataset, DataLoader
import numpy as np
from tsp_problem import TSPProblem, TSPBatch
from solvers.concorde_solver import ConcordeSolver


def generate_random_points(n, min_coord=-1, max_coord=1):
    """
    Generate n random 2D points within the specified coordinate range.

    :param n: Number of points to generate
    :param min_coord: Minimum coordinate value (default -1)
    :param max_coord: Maximum coordinate value (default 1)
    :return: numpy array of shape (n, 2) containing the random points
    """
    return np.random.uniform(min_coord, max_coord, size=(n, 2))


class TSPDataset(Dataset):
    def __init__(self, num_samples=100, min_points=5, max_points=20, seed=None):
        self.num_samples = num_samples
        self.min_points = min_points
        self.max_points = max_points
        self.seed = seed
        self.problems = self._generate_problems()

    def _generate_problems(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        problems = []
        for _ in range(self.num_samples):
            num_points = np.random.randint(self.min_points, self.max_points + 1)
            points = generate_random_points(num_points)
            problem = TSPProblem(points)
            solution, _, _ = ConcordeSolver().solve(problem)
            solution = solution.tolist() + [solution[0]]  # Return to the depot
            problem.set_solution(solution)
            problems.append(problem)
        return problems

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.problems[idx]


def get_tsp_dataloader(batch_size=32, num_samples=100, min_points=5, max_points=20, seed=None):
    dataset = TSPDataset(num_samples, min_points, max_points, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=TSPBatch.collate_fn)

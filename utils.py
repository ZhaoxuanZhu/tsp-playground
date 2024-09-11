import os
from solvers.transformer_solver import TransformerSolver
import torch
import logging

logger = logging.getLogger(__name__)


def get_device():
    if torch.cuda.is_available():
        logger.info("Using CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info("Using MPS")
        return torch.device("mps")
    else:
        logger.info("Using CPU")
        return torch.device("cpu")


def get_experiment_dir(experiment_name):
    return os.path.join("results", experiment_name)


def find_latest_checkpoint(experiment_dir):
    """
    Find the latest checkpoint in the checkpoint directory.

    Returns:
        str: The path to the latest checkpoint file, or None if no checkpoints are found.
    """
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    if not os.path.exists(checkpoints_dir):
        return None

    checkpoints = [f for f in os.listdir(checkpoints_dir) if f.startswith("model_epoch_") and f.endswith(".pth")]

    if not checkpoints:
        return None

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[2].split(".")[0]))
    return os.path.join(checkpoints_dir, latest_checkpoint)


def save_model(solver, epoch, experiment_dir):
    """
    Save the solver model to the specified path.

    Args:
        solver (TransformerSolver): The solver model to be saved.
        path (str): The file path where the model should be saved.
    """
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    path = os.path.join(checkpoints_dir, f"model_epoch_{epoch+1}.pth")
    torch.save(solver.state_dict(), path)
    logger.info(f"Model saved to {path}")

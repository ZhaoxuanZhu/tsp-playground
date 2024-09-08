import os
from solvers.transformer_solver import TransformerSolver
import torch
import logging

logger = logging.getLogger(__name__)


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


def build_model(checkpoint_path=None):
    # Initialize the TransformerSolver
    d_model = 64
    nhead = 2
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 256
    dropout = 0.1

    solver = TransformerSolver(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)

    if checkpoint_path:
        solver.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")

    # Log the number of parameters of the solver in millions
    logger.info(f"Number of parameters in the solver: {sum(p.numel() for p in solver.parameters()) / 1e6:.2f} million")

    return solver


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

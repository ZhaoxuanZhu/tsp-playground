import os
import torch
import logging
from data import get_tsp_dataloader

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


def get_train_val_dataloaders(num_samples=16, min_points=8, max_points=8, train_seed=42, val_seed=42, batch_size=16):
    # Create load_path based on the parameters
    train_load_path = f"data/train/problems_{num_samples}_{min_points}_{max_points}_{train_seed}.pkl"
    val_load_path = f"data/val/problems_{int(num_samples / 10)}_{min_points}_{max_points}_{val_seed}.pkl"

    # Get the training and validation dataloaders
    train_dataloader = get_tsp_dataloader(
        batch_size=batch_size,
        num_samples=num_samples,
        min_points=min_points,
        max_points=max_points,
        seed=train_seed,
        load_path=train_load_path,
        shuffle=True,
    )
    val_dataloader = get_tsp_dataloader(
        batch_size=batch_size,
        num_samples=int(num_samples / 10),
        min_points=min_points,
        max_points=max_points,
        seed=val_seed,
        load_path=val_load_path,
        shuffle=False,
    )

    return train_dataloader, val_dataloader

import argparse
from helpers import (
    find_latest_checkpoint,
    get_experiment_dir,
)
from utils import train, evaluate, rl_fine_tune_with_dpo
from builder import build_model
import logging
import os
import torch

# Set a fixed seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Set up logging
def setup_logging(experiment_dir, log_level):
    """
    Set up logging for the experiment.

    Args:
        experiment_dir (str): Directory to store experiment results and logs.
    """
    log_file = os.path.join(experiment_dir, "experiment.log")
    os.makedirs(experiment_dir, exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or evaluate the TSP solver")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate", "rl_fine_tune"],
        default="train",
        help="Select whether to train or evaluate the model (default: train)",
    )
    parser.add_argument("--experiment_name", type=str, default="default", help="Name of the experiment")
    parser.add_argument("--load_checkpoint", action="store_true", help="Whether to load the latest checkpoint")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of training epochs (default: 100)")
    parser.add_argument("--max_lr", type=float, default=1e-4, help="Maximum learning rate (default: 1e-3)")
    parser.add_argument("--use_scheduler", action="store_true", help="Whether to use the learning rate scheduler")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )

    args = parser.parse_args()

    experiment_dir = get_experiment_dir(args.experiment_name)

    if args.load_checkpoint:
        latest_checkpoint = find_latest_checkpoint(experiment_dir)
    else:
        latest_checkpoint = None

    if args.mode == "rl_fine_tune":
        experiment_dir = experiment_dir + "_rl_fine_tune"

    # Sets up logging for the evaluation process.
    logger = setup_logging(experiment_dir, args.log_level)

    # Builds the model.
    solver = build_model(logger, latest_checkpoint)

    if args.mode == "train":
        train(solver, experiment_dir, args.max_epochs, args.max_lr, args.use_scheduler, logger)
    elif args.mode == "evaluate":
        evaluate(solver, experiment_dir, logger)
    elif args.mode == "rl_fine_tune":
        rl_fine_tune_with_dpo(solver, experiment_dir, args.max_epochs, args.max_lr, args.use_scheduler, logger)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

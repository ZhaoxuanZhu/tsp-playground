import torch
from visualization import plot_tours, plot_loss_curve
from data import get_tsp_dataloader
import argparse
from utils import (
    find_latest_checkpoint,
    save_model,
    get_experiment_dir,
    get_device,
)
from builder import build_model, build_optimizer_and_scheduler
import logging
import os
import time


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


def calculate_avg_solution_tour_length(dataloader, device):
    """
    Calculate the average solution tour length from the validation dataloader.

    Args:
        dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device to run the calculation on.

    Returns:
        float: The average solution tour length.
    """
    total_solution_tour_length = 0.0
    num_tours = 0
    for batch in dataloader:
        batch = batch.to(device)
        for problem in batch:
            total_solution_tour_length += problem.solution_tour_length.item()
            num_tours += 1

    return total_solution_tour_length / num_tours


def train(solver, experiment_dir, num_epochs, max_lr, use_scheduler, logger):
    """
    Train the solver model on the TSP dataset.

    Args:
        solver (TransformerSolver): The solver model to be trained.
        experiment_dir (str): Directory to store experiment results and logs.
        num_epochs (int): Number of training epochs.
        max_lr (float): Maximum learning rate.
        use_scheduler (bool): Whether to use the learning rate scheduler.
        logger (logging.Logger): Logger for logging training information.
    """
    device = get_device()

    solver.to(device)

    # Get the training and validation dataloaders
    train_dataloader = get_tsp_dataloader(batch_size=16, num_samples=16, min_points=5, max_points=15, seed=42)
    val_dataloader = get_tsp_dataloader(batch_size=16, num_samples=16, min_points=5, max_points=15, seed=42)

    # Build the optimizer and scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        solver, max_lr, use_scheduler, len(train_dataloader), num_epochs
    )

    # Training loop
    losses = []
    val_losses = []
    val_tour_lengths = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        start_time = time.time()
        for batch in train_dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = solver(batch)
            loss.backward()
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            epoch_loss += loss.item()
        epoch_time = time.time() - start_time

        # Step the learning rate scheduler
        current_lr = scheduler.get_last_lr()[0] if scheduler else max_lr

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        losses.append(avg_epoch_loss)

        if (epoch + 1) % 100 == 0:
            avg_val_loss, avg_tour_length = validate(solver, val_dataloader, device)
            # Save the model checkpoint
            save_model(solver, epoch, experiment_dir)
            val_losses.append(avg_val_loss)
            val_tour_lengths.append(avg_tour_length)
            logger.info("-" * 50)  # This will log a line of 50 dashes as a separator
            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
            logger.info(f"Epoch Time: {epoch_time:.2f} seconds")
            logger.info(f"Average Train Loss: {avg_epoch_loss:.4f}")
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"Validation Tour Length: {avg_tour_length:.4f}")
            logger.info(f"Current learning rate: {current_lr:.6f}")

    # Calculate the average solution tour length from the validation dataloader
    avg_solution_tour_length = calculate_avg_solution_tour_length(val_dataloader, device)
    logger.info(f"Average Solution Tour Length: {avg_solution_tour_length:.4f}")

    # Add the average solution tour length to the list for plotting
    val_solution_tour_lengths = [avg_solution_tour_length] * len(val_tour_lengths)

    # Plot the loss curve
    plot_loss_curve(losses, val_losses, val_tour_lengths, val_solution_tour_lengths, experiment_dir)

    # Plot the tours
    solver.eval()
    tours = solver.solve(batch)
    plot_tours(batch, tours, experiment_dir)


def validate(solver, val_dataloader, device):
    """
    Validate the solver on the validation dataset.

    Args:
        solver (TransformerSolver): The solver model to be validated.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): The device to run the validation on.

    Returns:
        tuple: A tuple containing:
            - avg_val_loss (float): The average validation loss.
            - avg_tour_length (float): The average tour length on the validation set.
    """
    val_loss = 0.0
    total_tour_length = 0.0
    num_tours = 0
    solver.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for validation
        for batch in val_dataloader:
            batch = batch.to(device)
            loss = solver(batch)
            val_loss += loss.item()
            tours = solver.solve(batch)

            # Calculate tour lengths
            for i in range(batch.points.shape[0]):
                tour = tours[i]
                points = batch.points[i]
                tour_length = torch.sum(torch.norm(points[tour[1:]] - points[tour[:-1]], dim=1))
                total_tour_length += tour_length.item()
                num_tours += 1

    avg_val_loss = val_loss / len(val_dataloader)
    avg_tour_length = total_tour_length / num_tours
    solver.train()
    return avg_val_loss, avg_tour_length


def evaluate(solver, experiment_dir, logger):
    """
    Evaluate the solver on the validation dataset.

    Args:
        solver (TransformerSolver): The solver model to be evaluated.
        experiment_dir (str): Directory to store experiment results and logs.
    """

    # Determines the device (CPU/GPU) to run the evaluation on.
    device = get_device()

    solver.to(device)

    # Load the validation dataset
    val_dataloader = get_tsp_dataloader(batch_size=16, num_samples=16, min_points=8, max_points=8, seed=42)
    # Validate the solver
    avg_val_loss, avg_tour_length = validate(solver, val_dataloader, device)
    avg_solution_tour_length = calculate_avg_solution_tour_length(val_dataloader, device)
    logger.info(
        f"Validation Loss: {avg_val_loss:.4f}, Validation Tour Length: {avg_tour_length:.4f}, Validation Solution Tour Length: {avg_solution_tour_length:.4f}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train or evaluate the TSP solver")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "evaluate"],
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

    # Sets up logging for the evaluation process.
    logger = setup_logging(experiment_dir, args.log_level)

    if args.load_checkpoint:
        latest_checkpoint = find_latest_checkpoint(experiment_dir)
    else:
        latest_checkpoint = None

    solver = build_model(logger, latest_checkpoint)

    if args.mode == "train":
        train(solver, experiment_dir, args.max_epochs, args.max_lr, args.use_scheduler, logger)
    elif args.mode == "evaluate":
        evaluate(solver, experiment_dir, logger)

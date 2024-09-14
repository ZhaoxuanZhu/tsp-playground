import torch
import copy
import os
from helpers import save_model, get_device, get_train_val_dataloaders
from visualization import plot_tours, plot_loss_curve
from data import get_tsp_dataloader
from builder import build_optimizer_and_scheduler
import time
from tqdm import tqdm


def calculate_tour_lengths(batch, tours):
    """
    Calculate the lengths of tours for a batch.

    Args:
        batch (Batch): The batch of TSP instances.
        tours (torch.Tensor): The tours generated by the solver.

    Returns:
        torch.Tensor: The lengths of the tours.
    """
    tour_points = torch.gather(batch.points, 1, tours.clamp(min=0).unsqueeze(-1).expand(-1, -1, 2))
    tour_lengths = torch.sum(
        torch.norm(tour_points[:, 1:] - tour_points[:, :-1], dim=2) * batch.padding_mask.float(), dim=1
    )
    return tour_lengths


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
        solution_tours = batch.solutions
        tour_lengths = calculate_tour_lengths(batch, solution_tours)
        total_solution_tour_length += tour_lengths.sum().item()
        num_tours += len(tour_lengths)

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
    num_samples = 51200
    min_points = 10
    max_points = 15
    train_seed = 42
    val_seed = 21
    batch_size = 2048
    train_dataloader, val_dataloader = get_train_val_dataloaders(num_samples, min_points, max_points, train_seed, val_seed, batch_size)

    # Calculate the average solution tour length from the validation dataloader
    avg_solution_tour_length = calculate_avg_solution_tour_length(val_dataloader, device)
    logger.info(f"Average Solution Tour Length: {avg_solution_tour_length:.4f}")

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
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as pbar:
            for batch in train_dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = solver(batch)
                loss.backward()
                optimizer.step()
                if use_scheduler:
                    scheduler.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)  # Update the progress bar
        epoch_time = time.time() - start_time

        # Step the learning rate scheduler
        current_lr = scheduler.get_last_lr()[0] if scheduler else max_lr

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        losses.append(avg_epoch_loss)

        if (epoch + 1) % 100 == 0:
            avg_val_loss, avg_tour_length = validate(solver, val_dataloader, device, experiment_dir, epoch)
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
            logger.info(f"Average Solution Tour Length: {avg_solution_tour_length:.4f}")
            logger.info(f"Current learning rate: {current_lr:.6f}")

    # Add the average solution tour length to the list for plotting
    val_solution_tour_lengths = [avg_solution_tour_length] * len(val_tour_lengths)

    # Plot the loss curve
    plot_loss_curve(losses, val_losses, val_tour_lengths, val_solution_tour_lengths, experiment_dir)

    # Plot the tours
    solver.eval()
    tours = solver.solve(batch)
    plot_tours(batch, tours, experiment_dir)


def validate(solver, val_dataloader, device, experiment_dir, epoch):
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
        for batch in tqdm(val_dataloader, desc="Validating", leave=False):
            batch = batch.to(device)
            loss = solver(batch)
            val_loss += loss.item()
            tours = solver.solve(batch)

            # Calculate tour lengths using batch operations
            tour_points = torch.gather(batch.points, 1, tours.unsqueeze(-1).expand(-1, -1, 2))
            tour_lengths = torch.sum(
                torch.norm(tour_points[:, 1:] - tour_points[:, :-1], dim=2) * batch.padding_mask.float(), dim=1
            )

            # Apply padding mask to tour lengths
            total_tour_length += tour_lengths.sum().item()

            # Compute the number of valid tours
            num_tours += batch.padding_mask.size(0)

    avg_val_loss = val_loss / len(val_dataloader)
    avg_tour_length = total_tour_length / num_tours
    os.makedirs(os.path.join(experiment_dir,'visualization'), exist_ok=True)
    plot_tours(batch, tours, os.path.join(experiment_dir,'visualization', f'epoch_{epoch}_tour.png'))
    solver.train()
    return avg_val_loss, avg_tour_length


def rl_fine_tune_with_dpo(solver, experiment_dir, num_epochs, max_lr, use_scheduler, logger):
    """
    Perform RL fine-tuning on the solver using Direct Preference Optimization (DPO).

    Args:
        solver (TransformerSolver): The solver model to be fine-tuned.
        experiment_dir (str): Directory to store experiment results and logs.
        num_epochs (int): Number of fine-tuning epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        logger (logging.Logger): Logger for logging training information.
    """
    device = get_device()
    solver.to(device)
    # Make a copy of the solver
    pretrained_solver = copy.deepcopy(solver)
    pretrained_solver.to(device)
    for param in pretrained_solver.parameters():
        param.requires_grad = False

    # Get the training and validation dataloaders
    num_samples = 51200
    min_points = 10
    max_points = 15
    train_seed = 42
    val_seed = 21
    batch_size = 1024
    train_dataloader, val_dataloader = get_train_val_dataloaders(num_samples, min_points, max_points, train_seed, val_seed, batch_size)

    optimizer, scheduler = build_optimizer_and_scheduler(
        solver, max_lr, use_scheduler, len(train_dataloader), num_epochs
    )

    # Calculate the average solution tour length from the validation dataloader
    avg_solution_tour_length = calculate_avg_solution_tour_length(val_dataloader, device)
    logger.info(f"Average Solution Tour Length: {avg_solution_tour_length:.4f}")

    beta = 0.1

    for epoch in range(num_epochs):
        total_loss = 0.0
        start_time = time.time()
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False) as pbar:
            for batch in train_dataloader:
                batch = batch.to(device)

                # Generate two sets of tours
                with torch.no_grad():
                    tours_1 = solver.solve(batch, sample=True)
                    tours_2 = solver.solve(batch, sample=True)

                    # Calculate tour lengths
                    lengths_1 = calculate_tour_lengths(batch, tours_1)
                    lengths_2 = calculate_tour_lengths(batch, tours_2)

                    # Calculate log probabilities of both tours using the pretrained solver
                    pretrained_log_probs_1 = pretrained_solver.log_probability(batch, tours_1)
                    pretrained_log_probs_2 = pretrained_solver.log_probability(batch, tours_2)

                # Calculate log probabilities of both tours
                log_probs_1 = solver.log_probability(batch, tours_1)
                log_probs_2 = solver.log_probability(batch, tours_2)

                # Calculate DPO loss
                loss = -torch.mean(
                    torch.nn.functional.logsigmoid(
                        beta * (log_probs_1 - log_probs_2 - (pretrained_log_probs_1 - pretrained_log_probs_2))
                    )
                    * (lengths_2 > lengths_1).float()
                    + torch.nn.functional.logsigmoid(
                        beta * (log_probs_2 - log_probs_1 - (pretrained_log_probs_2 - pretrained_log_probs_1))
                    )
                    * (lengths_1 > lengths_2).float()
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

                total_loss += loss.item()
                epoch_time = time.time() - start_time
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)  # Update the progress bar

        # Step the learning rate scheduler
        current_lr = scheduler.get_last_lr()[0] if scheduler else max_lr

        # Calculate the average loss for the epoch
        avg_epoch_loss = total_loss / len(train_dataloader)

        # Validate and save model periodically
        if (epoch + 1) % 10 == 0:
            avg_val_loss, avg_tour_length = validate(solver, val_dataloader, device, experiment_dir, epoch)
            logger.info("-" * 50)  # This will log a line of 50 dashes as a separator
            logger.info(f"Epoch [{epoch+1}/{num_epochs}]")
            logger.info(f"Epoch Time: {epoch_time:.2f} seconds")
            logger.info(f"Average Train Loss: {avg_epoch_loss:.4f}")
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"Validation Tour Length: {avg_tour_length:.4f}")
            logger.info(f"Average Solution Tour Length: {avg_solution_tour_length:.4f}")
            logger.info(f"Current learning rate: {current_lr:.6f}")
            save_model(solver, epoch, experiment_dir)

    logger.info("RL fine-tuning with DPO completed.")


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
    num_samples = 51200
    min_points = 10
    max_points = 15
    val_seed = 21

    val_load_path = f"data/val/problems_{num_samples}_{min_points}_{max_points}_{val_seed}.pkl"
    val_dataloader = get_tsp_dataloader(
        batch_size=2048,
        num_samples=int(num_samples / 10),
        min_points=min_points,
        max_points=max_points,
        seed=val_seed,
        load_path=val_load_path,
    )
    # Validate the solver
    avg_val_loss, avg_tour_length = validate(solver, val_dataloader, device, experiment_dir, epoch=-1)
    avg_solution_tour_length = calculate_avg_solution_tour_length(val_dataloader, device)
    logger.info(
        f"Validation Loss: {avg_val_loss:.4f}, Validation Tour Length: {avg_tour_length:.4f}, Validation Solution Tour Length: {avg_solution_tour_length:.4f}"
    )

import matplotlib.pyplot as plt
import os


def plot_tours(batch_problems, tours, save_path, num_subfigures=8):
    columns_per_row = 4
    # Plot the tours in subfigures
    rows = (num_subfigures + columns_per_row - 1) // columns_per_row * 2
    cols = min(num_subfigures, columns_per_row)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))

    for i, (problem, tour) in enumerate(zip(batch_problems, tours)):
        if i >= num_subfigures:
            break

        points = problem.points
        solution = problem.solution

        # Plot the solution
        ax_solution = axes[i // columns_per_row * 2, i % columns_per_row]
        ax_solution.scatter(points[:, 0], points[:, 1], c="blue")

        solution_length = 0
        for j in range(len(solution) - 1):
            start = points[solution[j]]
            end = points[solution[j + 1]]
            ax_solution.plot([start[0], end[0]], [start[1], end[1]], c="green", alpha=0.5)
            ax_solution.text(start[0], start[1], f"{j}", color="green", fontsize=12, ha="left", va="top")
            solution_length += ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5

        start = points[solution[-1]]
        end = points[solution[0]]
        ax_solution.plot([start[0], end[0]], [start[1], end[1]], c="green", alpha=0.5)
        ax_solution.text(start[0], start[1], f"{len(solution)-1}", color="green", fontsize=12, ha="left", va="top")
        solution_length += ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5

        ax_solution.set_title(f"Tour {i+1} - Solution")
        ax_solution.set_xlabel("X coordinate")
        ax_solution.set_ylabel("Y coordinate")
        ax_solution.text(
            0.05,
            0.95,
            f"Length: {solution_length:.2f}",
            transform=ax_solution.transAxes,
            verticalalignment="top",
            fontsize=10,
        )

        # Plot the prediction
        ax_prediction = axes[i // columns_per_row * 2 + 1, i % columns_per_row]
        ax_prediction.scatter(points[:, 0], points[:, 1], c="blue")

        prediction_length = 0
        for j in range(len(tour) - 1):
            start = points[tour[j].item()]
            end = points[tour[j + 1].item()]
            ax_prediction.plot([start[0], end[0]], [start[1], end[1]], c="red", alpha=0.5)
            ax_prediction.text(start[0], start[1], str(j), color="red", fontsize=12, ha="right", va="bottom")
            prediction_length += ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5

        start = points[tour[-1].item()]
        end = points[tour[0].item()]
        ax_prediction.plot([start[0], end[0]], [start[1], end[1]], c="red", alpha=0.5)
        ax_prediction.text(start[0], start[1], str(len(tour) - 1), color="red", fontsize=12, ha="right", va="bottom")
        prediction_length += ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** 0.5

        ax_prediction.set_title(f"Tour {i+1} - Prediction")
        ax_prediction.set_xlabel("X coordinate")
        ax_prediction.set_ylabel("Y coordinate")
        ax_prediction.text(
            0.05,
            0.95,
            f"Length: {prediction_length:.2f}",
            transform=ax_prediction.transAxes,
            verticalalignment="top",
            fontsize=10,
        )

    for j in range(i + 1, num_subfigures):
        fig.delaxes(axes[j // columns_per_row * 2, j % columns_per_row])
        fig.delaxes(axes[j // columns_per_row * 2 + 1, j % columns_per_row])

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_curve(losses, val_losses, val_tour_lengths, val_solution_tour_lengths, experiment_dir):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))

    # Training Loss Curve
    ax1.plot(losses)
    ax1.set_title("Training Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    # Validation Loss Curve
    ax2.plot(val_losses)
    ax2.set_title("Validation Loss Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")

    # Validation Tour Length Curve
    ax3.plot(val_tour_lengths)
    ax3.plot(val_solution_tour_lengths)
    ax3.legend(["Validation Tour Length", "Validation Solution Tour Length"])
    ax3.set_title("Validation Tour Length Curve")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Tour Length")

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "loss_curves.png"), dpi=300, bbox_inches="tight")
    plt.close("all")  # Close all figures

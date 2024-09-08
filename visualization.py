import matplotlib.pyplot as plt
import os


def plot_tours(batch_problems, tours, experiment_dir):
    # Plot the tours in subfigures
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, (problem, tour) in enumerate(zip(batch_problems, tours)):
        if i >= 5:  # We only have 5 subfigures
            break

        ax = axes[i]
        points = problem.points

        # Plot the points
        ax.scatter(points[:, 0], points[:, 1], c="blue")

        # Plot the predicted tour
        for j in range(len(tour) - 1):
            start = points[tour[j].item()]
            end = points[tour[j + 1].item()]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                c="red",
                alpha=0.5,
                label="Predicted" if j == 0 else "",
            )
            # Add text for the order of the tour
            ax.text(
                start[0],
                start[1],
                str(j),
                color="red",
                fontsize=12,
                ha="right",
                va="bottom",
            )

        # Plot the return to start for predicted tour
        start = points[tour[-1].item()]
        end = points[tour[0].item()]
        ax.plot([start[0], end[0]], [start[1], end[1]], c="red", alpha=0.5)
        # Add text for the last point
        ax.text(
            start[0],
            start[1],
            str(len(tour) - 1),
            color="red",
            fontsize=12,
            ha="right",
            va="bottom",
        )

        # Plot the solution from batch_problem
        solution = problem.solution
        for j in range(len(solution) - 1):
            start = points[solution[j]]
            end = points[solution[j + 1]]
            ax.plot(
                [start[0], end[0]],
                [start[1], end[1]],
                c="green",
                alpha=0.5,
                label="Solution" if j == 0 else "",
            )
            # Add text for the order of the solution
            ax.text(
                start[0],
                start[1],
                f"{j}",
                color="green",
                fontsize=12,
                ha="left",
                va="top",
            )

        # Plot the return to start for solution
        start = points[solution[-1]]
        end = points[solution[0]]
        ax.plot([start[0], end[0]], [start[1], end[1]], c="green", alpha=0.5)
        # Add text for the last point of the solution
        ax.text(
            start[0],
            start[1],
            f"{len(solution)-1}",
            color="green",
            fontsize=12,
            ha="left",
            va="top",
        )

        ax.set_title(f"Tour {i+1}")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.legend()

    # Remove any unused subfigures
    for j in range(i + 1, 6):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(os.path.join(experiment_dir, "tours.png"))
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

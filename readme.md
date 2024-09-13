# TSP Solver with Transformer

This project implements a Transformer-based solver for the Traveling Salesman Problem (TSP).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/tsp-transformer-solver.git
   cd tsp-transformer-solver
   ```

2. Create and activate a conda environment:
   ```
   conda create -n tsp-playground python=3.9
   conda activate tsp-playground
   ```


3. Clone and install the Concorde TSP Solver:
   ```
   git clone https://github.com/jvkersch/pyconcorde.git
   cd pyconcorde
   pip install -e .
   cd ..
   ```

   Note: Concorde requires a C compiler. Make sure you have one installed on your system.

4. Install the required packages:
   ```
   pip install torch torchvision numpy matplotlib einops
   ```

## Training and Evaluation

To train the model with supervised learning:

1. Run the following command:
   ```
   python main.py --mode train --max_epochs 10000 --experiment_name your_experiment_name --max_lr 1e-3 --use_scheduler
   ```

   You can adjust the following parameters:
   - `--max_epochs`: Set the maximum number of training epochs
   - `--experiment_name`: Choose a name for your experiment
   - `--max_lr`: Set the maximum learning rate
   - `--use_scheduler`: Include this flag to use the learning rate scheduler

2. The training process will log information to the console and save checkpoints and visualizations in the experiment directory.

To evaluate the model:

1. After training, run the following command:
   ```
   python main.py --mode evaluate --experiment_name your_experiment_name --load_checkpoint
   ```

   Make sure to use the same `experiment_name` as used during training.

2. The evaluation results will be displayed in the console, showing the validation loss and tour length.


## Results
### Supervised Learning
With the following dataloader setup:

```
train_dataloader = get_tsp_dataloader(
    batch_size=2048,
    num_samples=51200,
    min_points=10,
    max_points=15,
    seed=42,
)
val_dataloader = get_tsp_dataloader(
    batch_size=2048,
    num_samples=5120,
    min_points=10,
    max_points=15,
    seed=21,
)
```

The following training curves were achieved after 1000 epochs of training:

![Training Results](rendering/loss_curves.png)

It looks like after 1000 epochs the model is unepxectedly overperforming the solution provided by the Concorde TSP Solver.

The tours from the trained model can be visualized as follows:

![TSP Tour](rendering/epoch_-1_tour.png)




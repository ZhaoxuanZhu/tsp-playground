import torch
from solvers.transformer_solver import TransformerSolver
import logging

logger = logging.getLogger(__name__)


def build_model(logger, checkpoint_path=None):
    # Initialize the TransformerSolver
    d_model = 128
    nhead = 4
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 512
    dropout = 0.1
    use_kv_cache = True

    solver = TransformerSolver(
        d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, use_kv_cache
    )

    if checkpoint_path:
        solver.load_state_dict(torch.load(checkpoint_path))
        logger.info(f"Model loaded from checkpoint: {checkpoint_path}")

    # Log the number of parameters of the solver in millions
    logger.info(f"Number of parameters in the solver: {sum(p.numel() for p in solver.parameters()) / 1e6:.2f} million")

    return solver


def build_optimizer_and_scheduler(solver, max_lr, use_scheduler, steps_per_epoch, num_epochs):
    optimizer = torch.optim.Adam(solver.parameters(), lr=max_lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=num_epochs
        )
    else:
        scheduler = None
    return optimizer, scheduler

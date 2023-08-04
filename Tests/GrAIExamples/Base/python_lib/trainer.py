import torch
from typing import Optional

from python_lib.cifar import (
    iter_CIFAR,
    next_tensor_CIFAR,
)
from python_lib.model import SimpleAutoEncoder


def train_simple_auto_encoder(
    batch_size: int,
    label: int
):
    """
    Build a simple auto encoder trainer.

    Parameters
    ----------
    batch_size: int
        The batch size.
    label: int
        The label we want the data associated to.

    Returns
    -------
    A trainer on a simple auto encoder model.
    """
    torch.manual_seed(42)
    model = SimpleAutoEncoder().cpu()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    iter_data = iter_CIFAR(
        train=True,
        batch_size=batch_size,
        label=label,
        shuffle=False
    )

    while True:
        samples = next_tensor_CIFAR(iter_data)
        x = model(samples)
        loss = criterion(x, samples)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        yield float(loss.detach().numpy())


def step_simple_auto_encoder(trainer) -> Optional[float]:
    """
    Compute next loss from the simple auto encoder trainer.

    Parameters
    ----------
    trainer
        The auto encoder trainer.

    Returns
    -------
    float
        The loss computed.
    """
    try:
        loss = next(trainer)
    except StopIteration:
        return None
    return loss

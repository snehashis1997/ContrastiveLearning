
import torch

def ohem_loss(loss, keep_ratio=0.25):
    """
    Online Hard Example Mining (OHEM) loss.

    Args:
        loss (torch.Tensor): Per-sample loss values (batch_size,).
        keep_ratio (float): Ratio of hardest examples to keep.

    Returns:
        torch.Tensor: Mean loss over hard examples.
    """
    batch_size = loss.size(0)
    num_hard_examples = int(batch_size * keep_ratio)

    # Sort loss values in descending order and select top-k
    sorted_loss, _ = torch.sort(loss, descending=True)
    hard_loss = sorted_loss[:num_hard_examples]

    return hard_loss.mean()

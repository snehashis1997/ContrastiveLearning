import torch

def hardest_triplet_mining(embeddings, labels, margin=0.2):
    """
    Select hardest positive and hardest negative for triplet loss.

    Args:
        embeddings (torch.Tensor): Feature embeddings (batch_size, embed_dim).
        labels (torch.Tensor): Class labels (batch_size,).
        margin (float): Margin for triplet loss.

    Returns:
        torch.Tensor: Triplet loss.
    """
    batch_size = embeddings.shape[0]

    # Compute pairwise distances
    pairwise_dist = torch.cdist(embeddings, embeddings, p=2)  # (batch_size, batch_size)

    loss = 0.0
    for i in range(batch_size):
        anchor_label = labels[i]

        # Select positive and negative indices
        positive_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
        negative_mask = (labels != anchor_label)

        if positive_mask.any() and negative_mask.any():
            # Hardest positive (max distance)
            hardest_positive = pairwise_dist[i, positive_mask].max()

            # Hardest negative (min distance)
            hardest_negative = pairwise_dist[i, negative_mask].min()

            # Compute triplet loss
            loss += torch.clamp(hardest_positive - hardest_negative + margin, min=0)

    return loss / batch_size  # Average over batch

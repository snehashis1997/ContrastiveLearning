import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
        
        Args:
            temperature (float): Scaling factor for similarity scores.
        """
        super().__init__()
        self.temperature = temperature  # Store temperature parameter for scaling similarities
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)  # Cosine similarity function

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss for a batch of embeddings.
        
        Args:
            z_i (torch.Tensor): Embeddings from first augmented view (batch_size, feature_dim)
            z_j (torch.Tensor): Embeddings from second augmented view (batch_size, feature_dim)

        Returns:
            torch.Tensor: NT-Xent loss
        """
        batch_size = z_i.shape[0]  # Get batch size

        # Normalize embeddings to unit sphere (ensuring unit length)
        z_i = F.normalize(z_i, p=2, dim=-1)  # (batch_size, feature_dim)
        z_j = F.normalize(z_j, p=2, dim=-1)  # (batch_size, feature_dim)

        # Concatenate positive pairs: [z_i, z_j] to form a single tensor
        z = torch.cat([z_i, z_j], dim=0)  # (2 * batch_size, feature_dim)

        # Compute cosine similarity matrix between all pairs
        # 8*32 === 8*32*1  1*8*32
        similarity_matrix = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0))  # (2 * batch_size, 2 * batch_size)
        # print(similarity_matrix.shape)

        # Mask out self-similarity (diagonal elements should not contribute to loss calculation)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)  # Create identity matrix as mask
        similarity_matrix.masked_fill_(mask, float('-inf'))  # Replace diagonal elements with -inf

        # Select positive pairs (each embedding's match is exactly `batch_size` indices away)
        pos_sim = torch.diag(similarity_matrix, batch_size)  # Extract diagonal elements at batch_size distance
        neg_sim = similarity_matrix  # All pairs, including negatives

        # Compute loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.temperature  # Apply temperature scaling
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=z.device)  # Labels: positive pairs are at index 0
        loss = F.cross_entropy(logits, labels)  # Compute contrastive loss using cross entropy

        return loss  # Return final loss value

loss_fn = NTXentLoss(temperature=0.1)  # Instantiate loss function

# Example embeddings (batch_size=4, feature_dim=32)
z_i = torch.randn(4, 32)  # Random embeddings for first augmented view
z_j = torch.randn(4, 32)  # Random embeddings for second augmented view

loss = loss_fn(z_i, z_j)  # Compute NT-Xent loss
print(f"NT-Xent Loss: {loss.item()}")  # Print the loss value

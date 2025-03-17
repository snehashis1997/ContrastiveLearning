# import torch
# import torch.nn.functional as F

# A = torch.randn(8, 32)  # Shape (8, 32, 1)
# B = torch.randn(8, 32)  # Shape (1, 8, 32)
# C = B.T
# # Reshape both tensors to match feature dimensions (32)
# # A = A.squeeze(-1)  # Now (8, 32)
# # B = B.squeeze(0)   # Now (8, 32)

# # Compute Cosine Similarity
# cos_sim = F.cosine_similarity(A.unsqueeze(1), B.unsqueeze(0), dim=-1)
# print(cos_sim.shape)  # Expected output: (8, 8)


# # A = torch.randn(8, 32)  # Shape (8, 32, 1)
# # B = torch.randn(32, 8)  # Shape (1, 8, 32)
# numerator = torch.mm(A, C)
# dinominetor = torch.norm(A, p=2, dim=1, keepdim=True) * torch.norm(C, p=2, dim=1, keepdim=True)

# print(numerator / dinominetor)

# import torch

# A = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# B = torch.tensor([[4, 5, 8], [9, 9, 8], [7, 7, 8]])

# # Row-wise dot product
# dot_product = torch.sum(A * B, dim=1)  # Sum across columns
# print(dot_product)  # Output: tensor([ 38, 129, 177])

# C = torch.mm(A, B.T)  # Matrix multiplication
# print(C)

# import torch
# import torch.nn.functional as F


# cosine_similarity_fn = torch.nn.CosineSimilarity(dim=-1)

# # Example tensors
# A = torch.randn(8, 32, 1)  # Shape (8, 32, 1)
# B = torch.randn(1, 8, 32)  # Shape (1, 8, 32)

# # Step 1: Remove singleton dimensions
# A = A.squeeze(-1)  # Now shape (8, 32)
# B = B.squeeze(0)   # Now shape (8, 32)

# C = A.clone()
# D = B.clone()

# # Step 2: Compute L2 norms
# A_norm = torch.norm(A, p=2, dim=1, keepdim=True)  # Shape (8, 1)
# B_norm = torch.norm(B, p=2, dim=1, keepdim=True)  # Shape (8, 1)

# # Step 3: Normalize the vectors
# A_normalized = A / (A_norm + 1e-8)  # Shape (8, 32)
# B_normalized = B / (B_norm + 1e-8)  # Shape (8, 32)

# # Step 4: Compute cosine similarity (matrix multiplication)
# cosine_similarity = A_normalized @ B_normalized.T  # (8, 32) @ (32, 8) → (8, 8)


# z_i = F.normalize(C, p=2, dim=-1)  # (batch_size, feature_dim)
# z_j = F.normalize(D, p=2, dim=-1)  # (batch_size, feature_dim)

# # Concatenate positive pairs: [z_i, z_j] to form a single tensor
# z = torch.cat([z_i, z_j], dim=0)  # (2 * batch_size, feature_dim)z_i = F.normalize(z_i, p=2, dim=-1)  # (batch_size, feature_dim)
# z_j = F.normalize(z_j, p=2, dim=-1)  # (batch_size, feature_dim)

# # Concatenate positive pairs: [z_i, z_j] to form a single tensor
# z = torch.cat([z_i, z_j], dim=0)  # (2 * batch_size, feature_dim)
# # print(z)
# similarity_matrix = cosine_similarity_fn(z.unsqueeze(1), z.unsqueeze(0))

# print(cosine_similarity.shape, similarity_matrix.shape)  # Output: torch.Size([8, 8])

import torch
import torch.nn.functional as F

# Example logits (before softmax)
logits = torch.tensor([[5.2, 2.3, -0.8], 
                       [4.9, 1.7,  0.1], 
                       [6.1, 3.0, -1.2]])

# Ground truth labels (correct class index is 0 for each row)
labels = torch.tensor([0, 0, 0])

# Step 1: Apply softmax along dim=1 (row-wise)
softmax_probs = F.softmax(logits, dim=1)

# Step 2: Select correct class probabilities using advanced indexing
correct_class_probs = softmax_probs[torch.arange(logits.shape[0]), labels]  # Extracts softmax scores for correct classes

# Step 3: Compute negative log-likelihood (cross-entropy manually)
loss = -torch.log(correct_class_probs)

# Step 4: Compute final mean loss (if batch-based)
final_loss = loss.mean()  # Taking mean over batch

print(f"Softmax Probabilities:\n{softmax_probs}")
print(f"Correct Class Probabilities: {correct_class_probs}")
print(f"Loss per Sample: {loss}")
print(f"Final Mean Loss: {final_loss}")
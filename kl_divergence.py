import torch
import torch.nn.functional as F

# kl_loss_fn(student_logits, teacher_logits, temperature=2.0)

# Example probability distributions (batch_size=1, num_classes=3)
P = torch.tensor([[0.1, 0.7, 0.2]])  # True distribution
Q = torch.tensor([[0.2, 0.5, 0.3]])  # Approximate distribution

# Convert to log probabilities (F.kl_div expects log-probabilities for Q)
Q_log = Q.log()

# Compute KL Divergence
kl_loss = F.kl_div(Q_log, P, reduction="batchmean")  # "batchmean" computes mean KL divergence across batch
print(f"KL Divergence: {kl_loss.item()}")
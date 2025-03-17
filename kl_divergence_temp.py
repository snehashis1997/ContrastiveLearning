import torch
import torch.nn.functional as F

logits = torch.tensor([[5.0, 2.0, -1.0]])

# Different temperatures
for T in [0.5, 1.0, 2.0, 5.0]:
    soft_probs = F.softmax(logits / T, dim=1)
    print(f"Temperature {T}:\n{soft_probs}\n")

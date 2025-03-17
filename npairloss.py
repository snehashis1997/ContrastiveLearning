import torch
import torch.nn.functional as F


class NPairLoss(torch.nn.Module):
    def __init__():
        pass
    def forward(self, positive_sample, negetive_sample, anchor):
        positive_sample = []
        
        

loss_fn = NPairLoss()  # Instantiate loss function

z_i = torch.randn(4, 32)
z_j = torch.randn(4, 32)

loss = loss_fn(z_i, z_j)
print(f"NT-Xent Loss: {loss.item()}")
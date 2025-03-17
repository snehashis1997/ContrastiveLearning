
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MLPHead(nn.Module):
    """
    A simple MLP projector/predictor head used in BYOL.
    """
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.layers(x)


class BYOL(nn.Module):
    """
    BYOL Model: Two identical networks (Online & Target), 
    where the target is an EMA (momentum-updated) version of the online network.
    """
    def __init__(self, backbone, feature_dim=256, moving_avg_decay=0.99):
        """
        Args:
            backbone: Any feature extractor (e.g., ResNet-50 without classifier).
            feature_dim: Output feature dimension for BYOL.
            moving_avg_decay: Decay rate for EMA update.
        """
        super().__init__()
        self.moving_avg_decay = moving_avg_decay

        # Online Network (Encoder + Projector + Predictor)
        self.online_encoder = backbone
        self.online_projector = MLPHead(backbone.out_dim, hidden_dim=4096, out_dim=feature_dim)
        self.online_predictor = MLPHead(feature_dim, hidden_dim=4096, out_dim=feature_dim)

        # Target Network (Momentum Updated)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)

        # Stop gradients in target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        """
        Compute BYOL loss using two augmented views.
        Args:
            x1, x2: Two different augmented views of the same image.
        Returns:
            BYOL loss.
        """
        # Online network forward pass
        z1 = self.online_predictor(self.online_projector(self.online_encoder(x1)))
        z2 = self.online_predictor(self.online_projector(self.online_encoder(x2)))

        # Target network forward pass (stop gradient)
        with torch.no_grad():
            t1 = self.target_projector(self.target_encoder(x1))
            t2 = self.target_projector(self.target_encoder(x2))

        # Compute BYOL loss
        loss = self.loss_fn(z1, t2) + self.loss_fn(z2, t1)
        return loss

    @staticmethod
    def loss_fn(p, z):
        """
        Computes BYOL loss using cosine similarity.
        Args:
            p: Online network predictions.
            z: Target network representations.
        Returns:
            Scalar loss value.
        """
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    @torch.no_grad()
    def update_moving_average(self):
        """
        EMA update for the target network parameters.
        """
        for online_params, target_params in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = self.moving_avg_decay * target_params.data + (1 - self.moving_avg_decay) * online_params.data

        for online_params, target_params in zip(
            self.online_projector.parameters(), self.target_projector.parameters()):
            target_params.data = self.moving_avg_decay * target_params.data + (1 - self.moving_avg_decay) * online_params.data


# Example Backbone (ResNet-50 without classifier)
class ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove classifier
        self.out_dim = 2048  # ResNet-50 final feature dimension

    def forward(self, x):
        return self.encoder(x).squeeze()


# Example Usage
backbone = ResNet50Backbone()
model = BYOL(backbone)

# Dummy Data (Batch of 4, 3-channel 224x224 images)
x1 = torch.randn(4, 3, 224, 224)
x2 = torch.randn(4, 3, 224, 224)

loss = model(x1, x2)
print(f"BYOL Loss: {loss.item()}")

# Update target network
model.update_moving_average()

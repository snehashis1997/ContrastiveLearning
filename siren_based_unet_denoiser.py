import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# SIREN Activation Function
class SineActivation(nn.Module):
    def __init__(self, omega=30):  # Omega controls frequency
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)

# SIREN-Based Convolution Block
class SirenConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.sine = SineActivation()

    def forward(self, x):
        return self.sine(self.conv(x))

# U-Net Encoder with SIREN activations
class SirenEncoder(nn.Module):
    def __init__(self, in_channels, features=[64, 128, 256, 512]):
        super().__init__()
        self.layers = nn.ModuleList()
        for feature in features:
            self.layers.append(SirenConvBlock(in_channels, feature))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = feature

    def forward(self, x):
        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)  # Store for skip connections
        return x, skips

# U-Net Decoder with SIREN activations
class SirenDecoder(nn.Module):
    def __init__(self, out_channels, features=[512, 256, 128, 64]):
        super().__init__()
        self.layers = nn.ModuleList()
        for feature in features:
            self.layers.append(nn.ConvTranspose2d(feature, feature // 2, kernel_size=2, stride=2))
            self.layers.append(SirenConvBlock(feature, feature // 2))

    def forward(self, x, skips):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i % 2 == 1:  # Apply skip connections
                x = torch.cat([x, skips[-(i // 2) - 1]], dim=1)
        return x

# Full U-Net with SIREN
class SirenUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = SirenEncoder(in_channels)
        self.decoder = SirenDecoder(out_channels)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x, skips = self.encoder(x)
        x = self.decoder(x, skips)
        return self.final_conv(x)

# Model Initialization
model = SirenUNet(in_channels=1, out_channels=1)
print(model)

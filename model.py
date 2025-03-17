from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool
from torch.nn import Linear
import torch
import torch_geometric.transforms as T

class Model(torch.nn.Module):
    def __init__(self, k=20, aggr='max'):
        """
        Initializes the contrastive learning model for point cloud data.
        
        Args:
            k (int): Number of nearest neighbors for DynamicEdgeConv.
            aggr (str): Aggregation method ('max', 'mean', 'sum') for edge convolution.
        """
        super().__init__()
        
        # Feature extraction using DynamicEdgeConv layers
        # Each point is represented by its k-nearest neighbors
        self.conv1 = DynamicEdgeConv(MLP([2 * 3, 64, 64]), k, aggr)  # Input: 3D coordinates
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)  # Feature extraction
        
        # Encoder head: Linear layer to process extracted features
        self.lin1 = Linear(128 + 64, 128)  # Combines features from both convolution layers
        
        # Projection head: Maps encoded features to a compact space for contrastive learning
        # Following the SimCLR paradigm for better feature separation
        self.mlp = MLP([128, 256, 32], norm=None)  
        
        # Data augmentations: Introduce variations for contrastive learning
        self.augmentation = T.Compose([T.RandomJitter(0.03), T.RandomFlip(1), T.RandomShear(0.2)])

    def forward(self, data, train=True):
        """
        Forward pass of the model.
        
        Args:
            data (torch_geometric.data.Data): Input point cloud batch.
            train (bool): If True, apply augmentations and return contrastive embeddings.
                         If False, return only the final feature representations.
        
        Returns:
            If train=True: Tuple (h_1, h_2, compact_h_1, compact_h_2) 
                          Representations before and after projection head for two augmentations.
            If train=False: Tensor with the final embeddings.
        """
        if train:
            # Apply augmentations twice to create two different views of the same data
            augm_1 = self.augmentation(data)
            augm_2 = self.augmentation(data)

            # Extract point positions and batch indices for both views
            pos_1, batch_1 = augm_1.pos, augm_1.batch
            pos_2, batch_2 = augm_2.pos, augm_2.batch

            # Process first augmented view through convolution layers
            x1 = self.conv1(pos_1, batch_1)
            x2 = self.conv2(x1, batch_1)
            h_points_1 = self.lin1(torch.cat([x1, x2], dim=1))

            # Process second augmented view through convolution layers
            x1 = self.conv1(pos_2, batch_2)
            x2 = self.conv2(x1, batch_2)
            h_points_2 = self.lin1(torch.cat([x1, x2], dim=1))
            
            # Aggregate global features using max pooling
            h_1 = global_max_pool(h_points_1, batch_1)
            h_2 = global_max_pool(h_points_2, batch_2)
        else:
            # In evaluation mode, process input without augmentations
            x1 = self.conv1(data.pos, data.batch)
            x2 = self.conv2(x1, data.batch)
            h_points = self.lin1(torch.cat([x1, x2], dim=1))
            return global_max_pool(h_points, data.batch)

        # Apply projection head to obtain contrastive embeddings
        compact_h_1 = self.mlp(h_1)
        compact_h_2 = self.mlp(h_2)

        return h_1, h_2, compact_h_1, compact_h_2

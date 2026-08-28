import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_shape = [1, 28, 28], hidden_dims = [256, 256], num_classes = 10):
        super().__init__()
        self.dims = hidden_dims
        # build MLP layers
        layers = []
        prev_dim = int(torch.prod(torch.tensor(input_shape)))
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        # build MLP model by stacking layers
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        batch_size = x.shape[0]
        return self.net(x.view(batch_size, -1))


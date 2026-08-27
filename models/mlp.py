import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim = 784, hidden_dims = [256, 256], num_classes = 10):
        super().__init__()
        
        # build MLP layers
        layers = []
        prev_dim = input_dim
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


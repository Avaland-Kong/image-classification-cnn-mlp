import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_shape = [1, 28, 28], channels = [32, 64], num_classes = 10):
        super().__init__()

        # check if the input shape is compatible with the number of pooling layers
        n_pool = len(channels)
        assert input_shape[1] >= 2 ** n_pool and input_shape[2] >= 2 ** n_pool, (
            f"Too many pooling layers ({n_pool}) for input spatial size "
            f"{input_shape[1]}x{input_shape[2]}; dimensions would become < 1"
        )
    
        self.dims = channels
        input_channels = input_shape[0]
        input_height = input_shape[1]
        input_width = input_shape[2]

        # create the feature layers of the CNN
        layers = []
        prev_channels = input_channels
        for channel in channels:
            layers.append(nn.Conv2d(prev_channels, channel, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(channel))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            prev_channels = channel
        self.features = nn.Sequential(*layers)

        # create the classifier
        out_height = input_height // 2 ** n_pool
        out_width = input_width // 2 ** n_pool
        flat_dim = channels[-1] * out_height * out_width
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
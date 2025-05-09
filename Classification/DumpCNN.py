import torch
import torch.nn as nn
import torch.nn.functional as F

class DumpCNN(nn.Module):
    def __init__(self, input_shape=(3, 64, 64), num_classes=3, use_dropout=True, dropout_rate=0.5, use_pool=True, num_conv_layers=3, hidden_dim=256):
        super().__init__()
        channels, height, width = input_shape

        self.hidden_dim = hidden_dim
        self.use_pool = use_pool
        self.use_dropout = use_dropout
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()

        in_channels = channels
        out_channels = 16

        for i in range(num_conv_layers):
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.bn_layers.append(nn.BatchNorm2d(out_channels))
            in_channels = out_channels
            out_channels *= 2
        
        if self.use_pool:
            self.pool = nn.MaxPool2d(2, 2)
        
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)

        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            out = self._forward_features(dummy_input)
            self.flattened_size = out.view(1, -1).shape[1]

        self.fc1 = nn.Linear(self.flattened_size, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, num_classes)

    def _forward_features(self, x):
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = F.relu(bn(conv(x)))
            if self.use_pool:
                x = self.pool(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        if self.use_dropout:
            x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

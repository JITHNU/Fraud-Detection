import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):  # match trained hidden size
        super().__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index=None):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1)

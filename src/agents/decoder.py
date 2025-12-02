# 文件路径: src/agents/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardDecoder(nn.Module):
    """
    Decoder to predict reward r_hat from (state, action, z)
    - state: state vector (dim state_dim)
    - action: action vector (dim action_dim)
    - z: environment embedding (dim z_dim)
    Returns scalar reward prediction.
    """

    def __init__(self, state_dim, action_dim, z_dim, hidden=300):
        super().__init__()
        input_dim = state_dim + action_dim + z_dim
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)
        self.relu = nn.ReLU()

        # weight init
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, state, action, z):
        """
        state: (batch, state_dim) tensor
        action: (batch, action_dim) tensor
        z: (batch, z_dim) tensor
        returns: (batch, 1) predicted reward
        """
        x = torch.cat([state, action, z], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        r_hat = self.head(x)
        return r_hat

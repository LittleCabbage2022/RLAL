import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# stable action/log-prob bounds
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6

def weights_init_(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, z_dim, action_dim, hidden1=300, hidden2=300):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + z_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc_mean = nn.Linear(hidden2, action_dim)
        self.fc_logstd = nn.Linear(hidden2, action_dim)
        self.relu = nn.ReLU()
        self.apply(weights_init_)

    def forward(self, state, z):
        """
        state: (B, state_dim)
        z: (B, z_dim)
        """
        x = torch.cat([state, z], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        return mu, log_std

    def sample(self, state, z):
        """
        Return: action ∈ [0,1], log_prob, mu
        action shape: (B, action_dim)
        log_prob shape: (B, 1)
        """
        mu, log_std = self.forward(state, z)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        # reparameterize
        z0 = dist.rsample()
        tanh_z = torch.tanh(z0)
        action = (tanh_z + 1.0) / 2.0  # map from (-1,1) to (0,1)
        # log prob correction for tanh
        log_prob = dist.log_prob(z0) - torch.log(1 - tanh_z.pow(2) + EPS)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, mu

    def sample_deterministic(self, state, z):
        mu, _ = self.forward(state, z)
        tanh_mu = torch.tanh(mu)
        action = (tanh_mu + 1.0) / 2.0
        return action

class QNetwork(nn.Module):
    def __init__(self, state_dim, z_dim, action_dim, hidden1=300, hidden2=300):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + z_dim + action_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.apply(weights_init_)

    def forward(self, state, z, action):
        x = torch.cat([state, z, action], dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q = self.fc3(x)
        return q
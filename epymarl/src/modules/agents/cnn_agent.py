import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------ PopArt Module ------------------
class PopArt(nn.Module):
    """
    PopArt layer for adaptive normalization of value outputs.
    This version is simplified.
    """

    def __init__(self, input_dim, output_dim=1, beta=0.99999, epsilon=1e-5):
        super(PopArt, self).__init__()
        self.mean = nn.Parameter(torch.zeros(output_dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(output_dim), requires_grad=False)
        self.beta = beta
        self.epsilon = epsilon
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        raw_value = self.linear(x)
        norm_value = (raw_value - self.mean) / (self.std + self.epsilon)
        return norm_value

    def update_stats(self, new_mean, new_std):
        self.mean.data = self.beta * self.mean.data + (1 - self.beta) * new_mean
        self.std.data = self.beta * self.std.data + (1 - self.beta) * new_std

    def unnormalize(self, norm_value):
        return norm_value * (self.std + self.epsilon) + self.mean


# ------------------ CPC Module ------------------
class CPC(nn.Module):
    """
    Contrastive Predictive Coding for auxiliary representation learning.
    This is a simplified placeholder.
    """

    def __init__(self, feature_dim=64, latent_dim=128):
        super(CPC, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, latent_dim),
            nn.ReLU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        latent = self.encoder(x)
        return latent

    def compute_cpc_loss(self, latent_features, future_features):
        predicted = self.predictor(latent_features)
        loss = F.mse_loss(predicted, future_features)
        return loss


# ------------------ Modified ACBAgent for Flat Observations ------------------
class ACBAgent(nn.Module):
    """
    Modified ACBAgent for environments with flat (vector) observations.
    CNN structure is removed if the input is flat.
    """

    def __init__(self, input_shape, args):
        super(ACBAgent, self).__init__()
        self.args = args
        self.n_actions = args.n_actions

        # Determine whether input is image-like or flat.
        # If input_shape is a tuple of length 3, we treat it as image; otherwise, as flat.
        if isinstance(input_shape, tuple):
            if len(input_shape) == 3:
                self.use_cnn = True
                self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=16,
                                       kernel_size=8, stride=8)
                self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                                       kernel_size=4, stride=1)
                dummy_input = torch.zeros(1, *input_shape)
                with torch.no_grad():
                    dummy_out = self.conv2(self.conv1(dummy_input))
                self.conv_out_dim = dummy_out.numel()
                self.mlp_fc1 = nn.Linear(self.conv_out_dim, 64)
            else:
                self.use_cnn = False
                # Assume flat observation: calculate product of dimensions
                flat_dim = int(np.prod(input_shape))
                self.mlp_fc1 = nn.Linear(flat_dim, 64)
        else:
            # input_shape is an int -> flat observation
            self.use_cnn = False
            flat_dim = input_shape
            self.mlp_fc1 = nn.Linear(flat_dim, 64)

        self.mlp_fc2 = nn.Linear(64, 64)

        # LSTM layer for temporal dependencies
        self.lstm = nn.LSTMCell(64, 128)

        # Policy head: output action logits
        self.policy_head = nn.Linear(128, self.n_actions)
        # Value head with PopArt for normalized value output
        self.value_head = PopArt(128, 1)

        # CPC module as auxiliary task (optional)
        self.use_cpc = getattr(args, "use_cpc", False)
        if self.use_cpc:
            self.cpc_module = CPC(feature_dim=64, latent_dim=128)
        else:
            self.cpc_module = None

        self._init_parameters()

    def _init_parameters(self):
        """
        Initialize convolutional and linear layers orthogonally.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def init_hidden(self, batch_size=1):
        """
        Initialize LSTM hidden states.
        Returns:
            A tuple (h, c) each of shape (batch_size, 128)
        """
        device = next(self.parameters()).device
        h = torch.zeros(batch_size, 128, device=device)
        c = torch.zeros(batch_size, 128, device=device)
        return (h, c)

    def forward(self, obs, hidden_state):
        # Process input using MLP for flat observations
        if self.use_cnn:
            x = F.relu(self.conv1(obs))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.mlp_fc1(x))
        else:
            x = F.relu(self.mlp_fc1(obs))
        x = F.relu(self.mlp_fc2(x))

        # Flatten the hidden state to 2D for LSTMCell
        h_in, c_in = hidden_state
        h_in = h_in.reshape(-1, h_in.size(-1))
        c_in = c_in.reshape(-1, c_in.size(-1))

        h_out, c_out = self.lstm(x, (h_in, c_in))

        # Reshape hidden states back to (batch_size, n_agents, hidden_dim)
        batch_size = int(obs.size(0) / self.args.n_agents)
        h_out = h_out.view(batch_size, self.args.n_agents, -1)
        c_out = c_out.view(batch_size, self.args.n_agents, -1)

        # Compute policy logits using the LSTM output (flatten back for the head)
        policy_logits = self.policy_head(h_out.view(-1, h_out.size(-1)))

        return policy_logits, (h_out, c_out)

    def compute_cpc_loss(self, features, future_features):
        """
        Compute the CPC auxiliary loss if enabled.

        Args:
            features (tensor): Intermediate MLP output (batch_size, 64)
            future_features (tensor): Target features for CPC.

        Returns:
            cpc_loss (tensor) or None.
        """
        if self.use_cpc and self.cpc_module is not None:
            latent = self.cpc_module(features)
            cpc_loss = self.cpc_module.compute_cpc_loss(latent, future_features)
            return cpc_loss
        return None

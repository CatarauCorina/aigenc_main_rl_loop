import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from baseline_models.masked_actions import CategoricalMasked
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)

        self.discrete_head = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, n_actions)
        )
        torch.nn.init.xavier_uniform_(self.discrete_head[-1].weight)
        self.discrete_head[-1].bias.data.fill_(0.01)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        if x.ndim == 4:  # raw image
            conv_out = self.conv(x).view(x.size()[0], -1)
        elif x.ndim == 2 and x.size(1) == 64:  # SETLE embedding
            conv_out = x
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")

        return self.discrete_head(conv_out)



class DQNSetle(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQNSetle, self).__init__()
        self.discrete_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, n_actions)
        )
        torch.nn.init.xavier_uniform_(self.discrete_head[-1].weight)
        self.discrete_head[-1].bias.data.fill_(0.01)

    def forward(self, x):
        if x.ndim == 3 and x.shape[1] == 1:
            x = x.squeeze(1)

        if x.ndim != 2 or x.size(1) != 64:
            raise ValueError(f"DQNSetle expects [B, 64] SETLE embeddings. Got {x.shape}")
        return self.discrete_head(x)

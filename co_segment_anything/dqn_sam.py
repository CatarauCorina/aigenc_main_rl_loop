import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from baseline_models.masked_actions import CategoricalMasked
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions_discrete, n_actions_continuous, env, masked=True):
        super(DQN, self).__init__()
        self.num_discrete_actions = n_actions_discrete
        self.num_continuous_actions = n_actions_continuous
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.discrete_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions_discrete)
        )

        # Continuous action head
        self.continuous_head = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions_continuous)
        )
        self.masked = masked
        self.env = env

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(*shape))
        return int(np.prod(o.size()))

    def forward(self, x, mask, inventory):
        conv_out = self.conv(x).view(x.size()[0], -1)
        discrete_action_probs = self.discrete_head(conv_out)
        # Continuous action prediction (no activation function for continuous values)
        continuous_action_params = F.tanh(self.continuous_head(conv_out))

        if self.masked:
            head = CategoricalMasked(logits=discrete_action_probs, mask=mask)
            q_values = head.probs
            value_of_max_action = [self.env.allowed_actions[q_val] for q_val in q_values.max(1)[1]]
            # Create a 1D list to store the indices
            actions = []
            # Find the indices where elements in 'value_of_max_action' match elements in 'a'
            for idx, max_action_value in enumerate(value_of_max_action):
                max_action_from_inventory = np.where(inventory[idx].squeeze(0) == max_action_value)[0].item()
                actions.append([max_action_from_inventory, continuous_action_params[idx][0].item(),
                                continuous_action_params[idx][1].item()])
        return q_values, continuous_action_params, actions

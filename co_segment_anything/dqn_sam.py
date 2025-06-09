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
        torch.nn.init.xavier_uniform_(self.discrete_head[2].weight)
        self.discrete_head[2].bias.data.fill_(0.01)

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
        if x.ndim == 4:  # image: [B, C, H, W]
            conv_out = self.conv(x).view(x.size()[0], -1)

        elif x.ndim == 2 and x.size(1) == 64:  # enriched SETLE vector
            conv_out = x
        else:
            raise ValueError(f"[⚠️] Unexpected input shape: {x.shape}")


        discrete_logits = self.discrete_head(conv_out)
        continuous_params = torch.tanh(self.continuous_head(conv_out))

        if (mask.sum(dim=1) == 0).any():
            print("🚨 MASK ERROR: At least one sample in batch has no valid actions!")
            print("Mask sums:", mask.sum(dim=1))

        # 🔍 Log raw logits stats before masking
        print(
            f"🔢 Raw logits stats - min: {discrete_logits.min().item():.4f}, max: {discrete_logits.max().item():.4f}, mean: {discrete_logits.mean().item():.4f}")

        if self.masked:
            # 🔍 Check mask stats
            print(f"🟩 Mask sum (allowed actions per sample): {[int(m.sum().item()) for m in mask]}")

            # Mask invalid actions
            mask = mask.squeeze(1)
            q_values = discrete_logits.masked_fill(~mask, float('-inf'))
            # Only consider valid Q-values (for logging/debug)
            valid_q_values = q_values.clone()
            valid_q_values[~mask] = 0
            mean_valid_q = valid_q_values.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            print(f"🟢 Valid Q-values mean across batch: {mean_valid_q.mean().item():.4f}")

            # # 🔍 Check post-mask q-values
            # print(
            #     f"🎯 Q-values after mask - min: {q_values.min().item():.4f}, max: {q_values.max().item():.4f}, mean: {q_values.mean().item():.4f}")

            best_action_indices = q_values.argmax(dim=1)  # index in allowed_actions

            # value_of_max_action = [self.env.allowed_actions[idx.item()] for idx in best_action_indices]
            value_of_max_action = [self.env.allowed_actions[int(idx)] for idx in best_action_indices]

            actions = []
            for idx, max_action_value in enumerate(value_of_max_action):
                inventory_idx = (inventory[idx].squeeze(0) == max_action_value).nonzero(as_tuple=False)
                inventory_idx = inventory_idx[0].item() if inventory_idx.numel() > 0 else 0

                actions.append([
                    inventory_idx,
                    continuous_params[idx][0].item(),
                    continuous_params[idx][1].item()
                ])

            return q_values, continuous_params, actions
        else:
            return discrete_logits, continuous_params, None




class DQNSetle(nn.Module):
    def __init__(self, input_shape, n_actions_discrete, n_actions_continuous, env, masked=True):
        super(DQNSetle, self).__init__()
        self.num_discrete_actions = n_actions_discrete
        self.num_continuous_actions = n_actions_continuous
        self.discrete_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions_discrete)
        )
        torch.nn.init.xavier_uniform_(self.discrete_head[2].weight)
        self.discrete_head[2].bias.data.fill_(0.01)

        # Continuous action head
        self.continuous_head = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions_continuous)
        )
        self.masked = masked
        self.env = env


    def forward(self, x, mask, inventory):
        # conv_out = self.conv(x).view(x.size()[0], -1)

        discrete_logits = self.discrete_head(x)
        continuous_params = torch.tanh(self.continuous_head(x))

        if mask.ndim > 2:
            mask = mask.squeeze()  #

        if (mask.sum(dim=1) == 0).any():
            print("🚨 MASK ERROR: At least one sample in batch has no valid actions!")
            print("Mask sums:", mask.sum(dim=1))

        # 🔍 Log raw logits stats before masking
        print(
            f"🔢 Raw logits stats - min: {discrete_logits.min().item():.4f}, max: {discrete_logits.max().item():.4f}, mean: {discrete_logits.mean().item():.4f}")

        if self.masked:
            # 🔍 Check mask stats
            print(f"🟩 Mask sum (allowed actions per sample): {[int(m.sum().item()) for m in mask]}")

            # Mask invalid actions
            mask = mask.squeeze(1)
            q_values = discrete_logits.masked_fill(~mask, float('-inf'))

            # Only consider valid Q-values (for logging/debug)
            valid_q_values = q_values.clone()
            valid_q_values[~mask] = 0
            mean_valid_q = valid_q_values.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            print(f"🟢 Valid Q-values mean across batch: {mean_valid_q.mean().item():.4f}")

            # # 🔍 Check post-mask q-values
            # print(
            #     f"🎯 Q-values after mask - min: {q_values.min().item():.4f}, max: {q_values.max().item():.4f}, mean: {q_values.mean().item():.4f}")

            best_action_indices = q_values.argmax(dim=1)  # index in allowed_actions

            # value_of_max_action = [self.env.allowed_actions[idx.item()] for idx in best_action_indices]
            value_of_max_action = [self.env.allowed_actions[int(idx)] for idx in best_action_indices]

            actions = []
            for idx, max_action_value in enumerate(value_of_max_action):
                inventory_idx = (inventory[idx].squeeze(0) == max_action_value).nonzero(as_tuple=False)
                inventory_idx = inventory_idx[0].item() if inventory_idx.numel() > 0 else 0

                actions.append([
                    inventory_idx,
                    continuous_params[idx][0].item(),
                    continuous_params[idx][1].item()
                ])

            return q_values, continuous_params, actions
        else:
            return discrete_logits, continuous_params, None



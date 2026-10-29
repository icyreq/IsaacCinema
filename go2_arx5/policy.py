"""Frozen two-headed actor and observation normalization; no PPO or optimizer."""
import torch
from torch import nn
from typing import Type

def build_mlp(layer_shape: tuple, activation, output_activation):
    layers = []
    for j in range(len(layer_shape) - 1):
        act = activation if j < len(layer_shape) - 2 else output_activation
        layers += [
            nn.Linear(layer_shape[j], layer_shape[j + 1]),
            act(),
        ]
    return nn.Sequential(*layers)


class TwoHeadedNet(nn.Module):
    def __init__(
        self,
        backbone_layers: list[int],
        branch_1_latent_layers: list[int],
        branch_2_latent_layers: list[int],
        branch_1_output_dim: int,
        branch_2_output_dim: int,
        intermediate_activation: Type[nn.Module],
    ):
        super().__init__()

        self.shared_backbone = build_mlp(
            layer_shape=backbone_layers, activation=intermediate_activation, output_activation=intermediate_activation
        )
        extended_branch_1_layers = [backbone_layers[-1]] + branch_1_latent_layers
        extended_branch_2_layers = [backbone_layers[-1]] + branch_2_latent_layers

        self.branch_1_latent = build_mlp(
            layer_shape=extended_branch_1_layers,
            activation=intermediate_activation,
            output_activation=intermediate_activation,
        )
        self.branch_2_latent = build_mlp(
            layer_shape=extended_branch_2_layers,
            activation=intermediate_activation,
            output_activation=intermediate_activation,
        )

        self.branch_1_output = nn.Linear(extended_branch_1_layers[-1], branch_1_output_dim)
        self.branch_2_output = nn.Linear(extended_branch_2_layers[-1], branch_2_output_dim)

    def forward(self, inputs: torch.Tensor):
        x = self.shared_backbone(inputs)
        x1 = self.branch_1_latent(x)
        x2 = self.branch_2_latent(x)
        out1 = self.branch_1_output(x1)
        out2 = self.branch_2_output(x2)

        return torch.cat([out1, out2], dim=-1)



class Controller:
    def __init__(self, checkpoint, config, device):
        weights = torch.load(checkpoint, map_location=device, weights_only=False)
        stats = weights["observation_preprocessor"]
        self.mean = stats["running_mean"].float()
        self.variance = stats["running_variance"].float()
        activation = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[config["activation"]]
        self.actor = TwoHeadedNet(
            [self.mean.numel(), *config["backbone"]], config["leg_branch"], config["arm_branch"],
            config["leg_actions"], config["arm_actions"], activation,
        ).to(device)
        self.actor.load_state_dict({k.removeprefix("actor."): v for k, v in weights["policy"].items() if k.startswith("actor.")})
        self.actor.eval().requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, observations):
        normalized = ((observations - self.mean) / (torch.sqrt(self.variance) + 1e-8)).clamp(-5., 5.)
        return self.actor(normalized)

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import mjx
from mjx import Agent, Action
from mortal_style_feature import MortalStyleFeature

class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=4):
        super().__init__()
        mid = max(channels // ratio, 16)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
        )

    def forward(self, x):
        avg_out = self.shared_mlp(x.mean(-1))
        max_out = self.shared_mlp(x.amax(-1))
        return (avg_out + max_out).sigmoid().unsqueeze(-1) * x

class ResBlock1D(nn.Module):
    def __init__(self, channels, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3,
                      padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01),
            nn.Mish(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01),
        )
        self.ca   = ChannelAttention(channels)
        self.actv = nn.Mish(inplace=True)

    def forward(self, x):
        return self.actv(self.ca(self.conv(x)) + x)

class MortalStyleNet(nn.Module):
    def __init__(self, in_channels=506, conv_channels=320,
                 num_blocks=24, n_actions=181):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(in_channels, momentum=0.01),
            nn.Mish(inplace=True),
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(conv_channels, momentum=0.01),
            nn.Mish(inplace=True),
        )

        dilations = [1 if i % 2 == 0 else 2 for i in range(num_blocks)]
        self.blocks = nn.Sequential(
            *[ResBlock1D(conv_channels, dilation=d) for d in dilations]
        )

        self.head = nn.Sequential(
            nn.Conv1d(conv_channels, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128, momentum=0.01),
            nn.Mish(inplace=True),
            nn.Conv1d(128, 32, kernel_size=1, bias=False),
            nn.BatchNorm1d(32, momentum=0.01),
            nn.Mish(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 34, 1024),
            nn.Mish(inplace=True),
            nn.Linear(1024, 512),
            nn.Mish(inplace=True),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

def load_mortalstyle(ckpt_path: str, n_actions: int = 181, device: str = "cpu") -> MortalStyleNet:
    model = MortalStyleNet(n_actions=n_actions)

    ckpt = torch.load(ckpt_path, map_location=device)

    if "state_dict" in ckpt:
        raw = ckpt["state_dict"]
        state_dict = {k.removeprefix("model."): v for k, v in raw.items()}
    elif "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    return model.to(device)

class MortalLikeAgent(Agent):
    def __init__(self, ckpt_path: str, device: str = "cpu") -> None:
        super().__init__()
        self.device = device
        self.model  = load_mortalstyle(ckpt_path, device=device)
        self.model.eval()

    def act(self, observation) -> Action:
        legal_actions = observation.legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]

        feature = MortalStyleFeature.produce(observation)

        with torch.no_grad():
            t_feat     = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_logit = self.model(t_feat)

        action_proba = torch.sigmoid(action_logit[0]).cpu().numpy()
        mask         = observation.action_mask()
        action_idx   = (mask * action_proba).argmax()

        return mjx.Action.select_from(action_idx, legal_actions)
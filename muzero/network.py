# muzero/network.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from muzero.config import Config


class Residual(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        h = F.relu(self.bn1(self.c1(x)))
        h = self.bn2(self.c2(h))
        return F.relu(x + h)


class MuZeroNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        ch = cfg.hidden_size

        # fₕ
        self.conv0 = nn.Conv2d(cfg.obs_shape[0], ch, 3, 1, 1)
        self.repr_stack = nn.Sequential(*[Residual(ch) for _ in range(3)])

        # (π, v)
        self.policy = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 4 * 4, ch),
            nn.ReLU(),
            nn.Linear(ch, cfg.action_space),
        )
        self.value = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ch * 4 * 4, ch),
            nn.ReLU(),
            nn.Linear(ch, 1),
            nn.Tanh(),
        )

        # g
        self.dyn_fc = nn.Linear(ch * 4 * 4 + cfg.action_space, ch * 4 * 4)
        self.dyn_reward = nn.Linear(ch * 4 * 4, 1)

    # ── API ───────────────────────────────────────────────────────────────
    def representation(self, obs: torch.Tensor) -> torch.Tensor:
        return self.repr_stack(F.relu(self.conv0(obs)))

    def prediction(self, h: torch.Tensor):
        return self.policy(h), self.value(h).squeeze(1)

    def dynamics(self, h: torch.Tensor, a_onehot: torch.Tensor):
        # concat latent + action
        inp = torch.cat([h.flatten(1), a_onehot], 1)
        h_next_flat = self.dyn_fc(inp)
        h_next = h_next_flat.view(-1, self.cfg.hidden_size, 4, 4)
        r = self.dyn_reward(h_next_flat).squeeze(1).tanh()
        return h_next, r


# ── quick self-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = Config()
    net = MuZeroNet(cfg)
    dummy = torch.randn(2, *cfg.obs_shape)
    h = net.representation(dummy)
    π, v = net.prediction(h)
    print("policy logits", π.shape, "value", v.shape)
    a = torch.eye(cfg.action_space)[[0, 1]]
    h2, r = net.dynamics(h, a)
    print("next h", h2.shape, "reward", r.shape)

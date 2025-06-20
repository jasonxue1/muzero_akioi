# muzero/replay_buffer.py
from __future__ import annotations
import random, collections, numpy as np, torch
from types import SimpleNamespace


class Game(dict):
    """obs, pi, value, reward"""


class ReplayBuffer:
    def __init__(self, cfg: SimpleNamespace):
        self.buf: collections.deque[Game] = collections.deque(maxlen=cfg.replay_size)
        self.cfg = cfg

    def add(self, game: Game):
        self.buf.append(game)

    def __len__(self):
        return len(self.buf)

    def sample(self, device):
        batch = random.sample(self.buf, self.cfg.batch_size)
        obs = np.stack([g["obs"][0] for g in batch])
        pi = np.stack([g["pi"][0] for g in batch])
        v = np.stack([g["value"][0] for g in batch])
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=device),
            torch.as_tensor(pi, dtype=torch.float32, device=device),
            torch.as_tensor(v, dtype=torch.float32, device=device),
        )

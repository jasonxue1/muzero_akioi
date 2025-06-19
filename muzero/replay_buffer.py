# muzero/replay_buffer.py
from __future__ import annotations
import random, collections, numpy as np, torch
from muzero.config import Config


class Game(dict):
    """obs, pi, value, reward"""


class ReplayBuffer:
    def __init__(self, cfg: Config):
        self.buf: collections.deque[Game] = collections.deque(maxlen=cfg.replay_size)
        self.cfg = cfg

    def add(self, game: Game):
        self.buf.append(game)

    def __len__(self):
        return len(self.buf)

    def sample(self, device="cpu"):
        batch = random.sample(self.buf, self.cfg.batch_size)
        obs = np.stack([g["obs"][0] for g in batch])
        pi = np.stack([g["pi"][0] for g in batch])
        v = np.stack([g["value"][0] for g in batch])
        return (
            torch.as_tensor(obs, dtype=torch.float32, device=device),
            torch.as_tensor(pi, dtype=torch.float32, device=device),
            torch.as_tensor(v, dtype=torch.float32, device=device),
        )


if __name__ == "__main__":
    cfg = Config(batch_size=4)
    rb = ReplayBuffer(cfg)
    dummy = np.zeros(cfg.obs_shape, np.float32)
    for _ in range(10):
        rb.add(Game(obs=[dummy], pi=[np.ones(4) / 4], value=[0.0], reward=[0.0]))
    o, p, v = rb.sample()
    print("sample shapes:", o.shape, p.shape, v.shape)

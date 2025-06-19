# env/akioi_env.py
from __future__ import annotations

import gymnasium as gym
import numpy as np
import akioi_2048


class Akioi2048Env(gym.Env):
    """
    Minimal Gym wrapper around the Rust backend.
    Observation: 17×4×4 one-hot (log₂ channels).
    Reward is scaled by 1/10 000 to keep MuZero targets ~O(1).
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(0, 1, (17, 4, 4), np.float32)
        self.board = akioi_2048.init()

    # ── helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _encode(board: list[list[int]]) -> np.ndarray:
        obs = np.zeros((17, 4, 4), dtype=np.float32)
        for r in range(4):
            for c in range(4):
                v = board[r][c]
                if v:
                    obs[int(np.log2(abs(v))), r, c] = 1.0
        return obs

    def _terminal(self) -> bool:
        # 无任何合法动作即终局
        return all(akioi_2048.step(self.board, a)[2] != 0 for a in range(4))

    # ── Gym API ──────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.board = akioi_2048.init()
        return self._encode(self.board), {}

    def step(self, action: int):
        self.board, reward, _ = akioi_2048.step(self.board, int(action))
        done = self._terminal()
        return self._encode(self.board), reward / 10_000.0, done, False, {}

    def render(self, mode="ansi"):
        return "\n".join(" ".join(f"{v:6d}" for v in row) for row in self.board)


# ── quick self-test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    env = Akioi2048Env()
    obs, _ = env.reset()
    print("obs shape", obs.shape)
    total = 0
    for _ in range(10):
        obs, r, done, *_ = env.step(env.action_space.sample())
        total += r
        if done:
            break
    print("10 random moves finished, reward =", total)

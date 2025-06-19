# muzero/config.py
from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    # ───────────────────────── 自博弈整体规模 / Self-Play Scope ─────────────────────────
    total_games: int = 100  # 训练总局数（默认 10 000）
    # Total self-play games to run (default: 10 000)

    log_interval: int = 50  # 每隔多少局打印一次日志（默认 50）
    # Games between console logs (default: 50)

    save_interval: int = 100  # 每隔多少局保存一次模型；0 = 不保存（默认 1 000）
    # Games between checkpoints; 0 = disable (default: 1 000)

    resume_path: str | None = None  # 断点续训：传 ckpt 文件路径，或 None 表示全新训练

    checkpoint_dir: str = "checkpoints"  # 模型保存目录

    # ─────────────────────────────── MCTS 参数 / MCTS ───────────────────────────────
    num_simulations: int = 50  # 每个动作的搜索模拟次数（默认 50）
    # Number of simulations per move (default: 50)

    discount: float = 1.0  # 奖励折扣 γ；2048 累加计分 → 固定 1.0
    # Reward discount γ; additive scoring ⇒ 1.0

    dirichlet_alpha: float = 0.3  # 根节点 Dirichlet 噪声 α（默认 0.3）
    # Dirichlet-noise alpha at root (default: 0.3)

    exploration_frac: float = 0.25  # 噪声混合比例 ε（默认 0.25）
    # Noise mix-in fraction ε (default: 0.25)

    # ─────────────────────── 网络规模 & 环境 / Network & Environment ───────────────────────
    obs_shape: Tuple[int, int, int] = (17, 4, 4)  # 观测张量尺寸（默认 (17,4,4)）
    # Observation tensor shape (default)

    hidden_size: int = 128  # 潜在表示通道数（默认 128）
    # Latent / residual channel size (default: 128)

    action_space: int = 4  # 动作空间大小：上下左右四方向（默认 4）
    # Action space size: 4 moves (default: 4)

    # ──────────────────────────── 训练与回放 / Training & Replay ────────────────────────────
    replay_size: int = 200_000  # 重放缓存容量（默认 200 000）
    # Replay buffer capacity (default: 200 000)

    batch_size: int = 256  # 每次梯度更新的样本数（默认 256）
    # Mini-batch size per optimisation step (default: 256)

    lr_init: float = 5e-4  # Adam 初始学习率（默认 5 × 10⁻⁴）
    # Initial learning rate for Adam (default: 5e-4)

    update_per_game: int = 50  # 每局自博弈后执行的优化步数（默认 50）
    # Optimisation steps after each game (default: 50)


if __name__ == "__main__":
    print("Default config:\n", Config())

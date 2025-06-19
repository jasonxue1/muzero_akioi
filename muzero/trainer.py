# muzero/trainer.py
# Trainer with timestamped checkpoints and a `latest.pt` symlink/alias.
# Author: Jason Xue (modified)

from __future__ import annotations

import os
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import trange

from env.akioi_env import Akioi2048Env
from muzero.config import Config
from muzero.network import MuZeroNet
from muzero.mcts import Node, run_mcts, add_noise
from muzero.replay_buffer import Game, ReplayBuffer


def play_one(env: Akioi2048Env, net: MuZeroNet, cfg: Config, device: str) -> Game:
    obs, _ = env.reset()
    g: Game = {
        "obs": [obs],
        "pi": [],
        "value": [],
        "reward": [],
        "total_reward": 0.0,
        "steps": 0,
    }

    root = Node(1.0)
    root.latent = net.representation(torch.tensor(obs[None], device=device))
    run_mcts(root, net, cfg, device)
    add_noise(root, cfg)

    done = False
    while not done:
        visits = np.array([c.visit for c in root.children.values()], np.float32)
        if visits.sum() == 0:
            visits = np.array([c.prior for c in root.children.values()], np.float32)
        visits /= visits.sum() + 1e-8
        action = int(np.random.choice(len(visits), p=visits))

        obs, r, done, *_ = env.step(action)
        g["pi"].append(visits)
        g["reward"].append(r)
        g["value"].append(root.value)
        g["total_reward"] += r
        g["steps"] += 1
        g["obs"].append(obs)

        child = root.children[action]
        if child.latent is None:
            a_one = torch.eye(cfg.action_space, device=device)[action].unsqueeze(0)
            child.latent, _ = net.dynamics(root.latent, a_one)
        root = child
        if not root.children:
            run_mcts(root, net, cfg, device)

    return g


def train(cfg: Config = Config()) -> None:
    # ❶ 设备选择
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        device = "mps"
    else:
        device = "cpu"
    print("Running on", device)

    # ❷ 组件初始化
    net = MuZeroNet(cfg).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=cfg.lr_init)
    env = Akioi2048Env()
    rb = ReplayBuffer(cfg)
    ckdir = Path(cfg.checkpoint_dir)
    ckdir.mkdir(exist_ok=True)

    # ❸ 断点续训
    start_g = 0
    if cfg.resume_path:
        ckpt = torch.load(cfg.resume_path, map_location=device)
        net.load_state_dict(ckpt["net"])
        optim.load_state_dict(ckpt["optim"])
        start_g = ckpt.get("game_idx", 0)
        print(f"✓ Resumed from {cfg.resume_path} (starting at game {start_g})")

    # ❹ 训练循环
    recent_scores, recent_steps = [], []
    for g_idx in trange(start_g, cfg.total_games, desc="games"):
        game = play_one(env, net, cfg, device)
        rb.add(game)
        recent_scores.append(game["total_reward"])
        recent_steps.append(game["steps"])
        if len(recent_scores) > cfg.log_interval:
            recent_scores.pop(0)
            recent_steps.pop(0)

        # 网络更新
        if len(rb) >= cfg.batch_size:
            for _ in range(cfg.update_per_game):
                obs, pi, v = rb.sample(device)
                h = net.representation(obs)
                logits, v_pred = net.prediction(h)
                loss = torch.nn.functional.cross_entropy(
                    logits, pi
                ) + torch.nn.functional.mse_loss(v_pred, v)
                optim.zero_grad()
                loss.backward()
                optim.step()

        # 日志
        if (g_idx + 1) % cfg.log_interval == 0:
            avg_s = np.mean(recent_scores) * 10
            avg_t = np.mean(recent_steps)
            loss_str = f"{float(loss):.4f}" if "loss" in locals() else "-"
            print(
                f"[{g_idx + 1:>6}] buf={len(rb):6d}"
                f"  loss={loss_str:>6}"
                f"  avgScore={avg_s:8.2f}  avgSteps={avg_t:6.1f}"
            )

        # 保存 checkpoint —— 带时间戳 & 更新 latest.pt
        if cfg.save_interval and (g_idx + 1) % cfg.save_interval == 0:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            fname = f"muzero_{g_idx + 1:06d}_{ts}.pt"
            path = ckdir / fname
            torch.save(
                {
                    "net": net.state_dict(),
                    "optim": optim.state_dict(),
                    "cfg": cfg,
                    "game_idx": g_idx + 1,
                },
                path,
            )
            print("✓ checkpoint saved →", path)

            # 更新 latest.pt 指向最新
            latest = ckdir / "latest.pt"
            try:
                if latest.exists() or latest.is_symlink():
                    latest.unlink()
                # 在同一目录下创建相对符号链接
                latest.symlink_to(fname)
            except (AttributeError, OSError):
                # 无法创建 symlink 时退回到复制
                shutil.copy(path, latest)
            print("  → latest.pt updated")


if __name__ == "__main__":
    # 全部配置集中在这里
    cfg = Config(
        total_games=10_000,
        log_interval=50,
        save_interval=1_000,
        num_simulations=50,
        hidden_size=128,
        batch_size=256,
        resume_path=None,  # 或 "checkpoints/latest.pt" / 具体 ckpt
    )
    train(cfg)

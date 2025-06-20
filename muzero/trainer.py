from __future__ import annotations

import os
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import trange
import polars as pl

from env.akioi_env import Akioi2048Env
from muzero.network import MuZeroNet
from muzero.mcts import Node, run_mcts, add_noise
from muzero.replay_buffer import Game, ReplayBuffer
from muzero.eval import eval

import printer

# 在文件开头 import
import statistics


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


def train(cfg: Config) -> None:
    last_loss = float("nan")
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
    ckdir.mkdir(parents=True, exist_ok=True)

    # ❸ 断点续训
    start_g = 0

    resume_path = Path(f"{cfg.checkpoint_dir}/latest.pt")
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)

        net.load_state_dict(ckpt["net"])
        optim.load_state_dict(ckpt["optim"])
        start_g = ckpt.get("game_idx", 0)
        print(f"✓ Resumed from {resume_path} (starting at game {start_g})")
        # restore replay buffer so batch_size is already met
        if "replay" in ckpt:
            rb.buf.extend(ckpt["replay"])
            print(f"✓ Restored {len(rb)} games into replay buffer")

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
                last_loss = float(loss)
                optim.zero_grad()
                loss.backward()
                optim.step()

        # 日志
        if (g_idx + 1) % cfg.log_interval == 0:
            avg_s = np.mean(recent_scores) * 10_000
            avg_t = np.mean(recent_steps)
            loss_str = f"{last_loss:.4f}"
            print(
                f"[{g_idx + 1:>6}] buf={len(rb):6d}"
                f"  loss={loss_str:>6}"
                f"  avgScore={avg_s:8.2f}  avgSteps={avg_t:6.1f}"
            )

        # 测试
        # if (g_idx + 1) % cfg.eval_interval == 0:
        #     avg_eval = evaluate(net, env, cfg, device, n_games=20)
        #     print(f"🎯 [Eval @ {g_idx + 1}] avg-2048-score={avg_eval:.1f}")

        # 保存 checkpoint —— 带时间戳 & 更新 latest.pt & test
        if cfg.save_interval and (g_idx + 1) % cfg.save_interval == 0:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            fname = f"muzero_{g_idx + 1:09d}_{ts}.pt"
            path = ckdir / fname
            torch.save(
                {
                    "net": net.state_dict(),
                    "optim": optim.state_dict(),
                    "cfg": cfg,
                    "game_idx": g_idx + 1,
                    "replay": list(rb.buf),
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
            print(f"✓ {latest} updated")
            if cfg.eval:
                eval_res = eval(cfg.eval_interval, f"{cfg.checkpoint_dir}/latest.pt")
                eval_special_res = [
                    min(eval_res),
                    max(eval_res),
                    sum(eval_res) / len(eval_res),
                ]
                eval_table = [["min", "max", "average"], eval_special_res]
                printer.print_table(eval_table)

                # 保存至csv
                csv_path = Path(cfg.csv_path)
                new_df = pl.DataFrame(
                    {
                        "id": [g_idx],
                        "min": [eval_special_res[0]],
                        "max": [eval_special_res[1]],
                        "average": [eval_special_res[2]],
                        "loss": [last_loss],
                    }
                )
                if csv_path.exists():
                    old_df = pl.read_csv(csv_path)
                    combined = pl.concat([old_df, new_df])
                else:
                    combined = new_df
                combined.write_csv(csv_path)
                print(f"✓ {csv_path} updated")


if __name__ == "__main__":
    # 全部配置集中在这里
    from muzero.config import Config

    cfg = Config()
    train(cfg)

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import akioi_2048

from muzero.config import Config
from muzero.network import MuZeroNet
from muzero.mcts import Node, run_mcts


def _encode_board(board: List[List[int]]) -> np.ndarray:
    obs = np.zeros((17, 4, 4), dtype=np.float32)
    for r in range(4):
        for c in range(4):
            v = board[r][c]
            if v:
                obs[int(math.log2(abs(v))), r, c] = 1.0
    return obs


def _is_valid_move(board: List[List[int]], action: int) -> bool:
    new_board, _, msg = akioi_2048.step(board, action)
    return msg == 0 and new_board != board


class MoveChooser:
    def __init__(
        self,
        checkpoint_path: str,
        mode: str = "mcts",
        device: Optional[str] = None,
        mcts_simulations: Optional[int] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        ckpt = torch.load(Path(checkpoint_path), map_location=device)
        self.cfg: Config = ckpt.get("cfg", Config())
        if mcts_simulations is not None:
            self.cfg.num_simulations = mcts_simulations

        self.net = MuZeroNet(self.cfg).to(device)
        self.net.load_state_dict(ckpt["net"])
        self.net.eval()

        self.mode = mode.lower()
        if self.mode not in ("policy", "mcts"):
            raise ValueError("mode must be 'policy' or 'mcts'")

    def choose(self, board: List[List[int]]) -> int:
        """
        对一个 4×4 board 输出最佳方向（0=Down,1=Right,2=Up,3=Left），
        必定返回一个合法方向，不会返回 None。
        """
        # 1) 计算动作优先序
        if self.mode == "policy":
            obs = _encode_board(board)
            obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
            with torch.inference_mode():
                h = self.net.representation(obs_t)
                logits, _ = self.net.prediction(h)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            order = list(np.argsort(-probs))
        else:  # mode == "mcts"
            obs = _encode_board(board)
            obs_t = torch.tensor(obs[None], dtype=torch.float32, device=self.device)
            root = Node(1.0)
            with torch.inference_mode():
                root.latent = self.net.representation(obs_t)
            run_mcts(root, self.net, self.cfg, self.device)
            visits = np.array(
                [root.children[a].visit for a in range(self.cfg.action_space)],
                dtype=np.float32,
            )
            order = list(np.argsort(-visits))

        # 2) 优先尝试合法移动
        for act in order:
            if _is_valid_move(board, act):
                return int(act)

        # 3) 如果没有找到合法移动，仍返回最高优先级方向
        #    （此时游戏应当结束，但我们保证有返回值）
        return int(order[0])

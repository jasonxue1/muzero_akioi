# muzero/mcts.py
from __future__ import annotations
import math, torch, numpy as np
from types import SimpleNamespace


class Node:
    __slots__ = ("prior", "visit", "value_sum", "reward", "children", "latent")

    def __init__(self, prior: float):
        self.prior = prior
        self.visit = 0
        self.value_sum = 0.0
        self.reward = 0.0
        self.children: dict[int, Node] = {}
        self.latent: torch.Tensor | None = None

    @property
    def value(self):
        return 0.0 if self.visit == 0 else self.value_sum / self.visit


def _ucb(parent: Node, child: Node, c1=1.25):
    return child.value + c1 * child.prior * math.sqrt(parent.visit) / (1 + child.visit)


def select(node: Node):
    return max(node.children.items(), key=lambda kv: _ucb(node, kv[1]))


@torch.no_grad()
def run_mcts(
    root: Node,
    net,
    cfg: SimpleNamespace,
):
    for _ in range(cfg.num_simulations):
        node, path = root, [root]
        # selection
        while node.children:
            a, node = select(node)
            path.append(node)

        # expansion
        latent = node.latent
        if latent is None:
            continue
        logits, v = net.prediction(latent)
        probs = torch.softmax(logits, 1)[0].cpu().numpy()
        for a, p in enumerate(probs):
            node.children[a] = Node(float(p))
        # back-prop
        _backprop(path, v.item(), cfg.discount)


def _backprop(path, v, γ):
    for node in reversed(path):
        node.visit += 1
        node.value_sum += v
        v = node.reward + γ * v


def add_noise(root: Node, cfg: SimpleNamespace):
    actions = list(root.children.keys())
    if not actions:
        return
    noise = np.random.dirichlet([cfg.dirichlet_alpha] * len(actions))
    for a, n in zip(actions, noise):
        child = root.children[a]
        child.prior = (
            child.prior * (1 - cfg.exploration_frac) + n * cfg.exploration_frac
        )

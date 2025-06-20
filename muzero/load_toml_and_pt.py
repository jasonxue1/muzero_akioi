from types import SimpleNamespace

# 优先使用 Python 3.11+ 内置 tomllib，否则退回到第三方 toml 包
import tomllib

import torch
from pathlib import Path


def load_toml(toml_path: Path) -> SimpleNamespace:
    toml_dict = tomllib.loads(toml_path.read_text())
    ns = SimpleNamespace()
    for k, v in toml_dict.items():
        setattr(ns, k, v)
    return ns


def load_pt_cfg(pt_path: Path) -> SimpleNamespace:
    checkpoint = torch.load(pt_path, map_location="cpu")
    ns = checkpoint["cfg"]
    return ns

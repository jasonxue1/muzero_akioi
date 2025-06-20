import os
from types import SimpleNamespace

# 优先使用 Python 3.11+ 内置 tomllib，否则退回到第三方 toml 包
import tomllib

import torch
from pathlib import Path


def load_toml_and_pt(toml_path: Path, pt_path: Path) -> SimpleNamespace:
    """
    读取 config.toml 和 model.pt（只取 checkpoint['cfg']），
    并将两部分内容合并到一个 SimpleNamespace 中。

    Args:
        toml_path: .toml 配置文件路径
        pt_path:   .pt 检查点文件路径（需包含 'cfg' 键）

    Returns:
        SimpleNamespace:
          - .toml 文件顶层键值作为 namespace 属性
          - checkpoint['cfg']（如果是 dict，则其键值也展开成属性；否则作为 .cfg 属性）
    """
    # 1. 加载 toml
    toml_dict = tomllib.loads(toml_path.read_text())

    # 2. 加载 pt 并只取 'cfg'
    checkpoint = torch.load(pt_path, map_location="cpu")
    pt_cfg = checkpoint["cfg"]

    # 3. 合并到 SimpleNamespace
    ns = SimpleNamespace()
    # 展开 toml 配置
    for k, v in toml_dict.items():
        setattr(ns, k, v)
    # 展开 pt_cfg
    if isinstance(pt_cfg, dict):
        for k, v in pt_cfg.items():
            setattr(ns, k, v)
    else:
        # 如果 pt['cfg'] 不是 dict，就当做一个整体对象保留
        ns.cfg = pt_cfg

    return ns


# —— 使用示例 ——
if __name__ == "__main__":
    cfg = load_toml_and_pt(
        Path("train_config.toml"), Path("models/test/checkpoints/latest.pt")
    )
    print(cfg)  # 你会在输出中看到所有合并后的属性

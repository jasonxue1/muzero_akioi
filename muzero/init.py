import tomllib  # Python 3.11+
from pathlib import Path
from datetime import datetime
import types
import torch

from muzero.network import MuZeroNet


def gen_init_ckpt(model_name: str):
    # 1) 读取 init_config.toml
    cfg_data = tomllib.loads(Path("init_config.toml").read_text())
    # toml 列表 → Python 元组
    cfg_data["obs_shape"] = tuple(cfg_data["obs_shape"])

    # 2) 构造一个简易的 config 对象（无需 import muzero.config）
    cfg = types.SimpleNamespace(**cfg_data)
    # 补齐 __post_init__ 会做的逻辑
    cfg.checkpoint_dir = f"models/{model_name}/checkpoints"

    # 3) 初始化网络和优化器
    net = MuZeroNet(cfg)
    optim = torch.optim.Adam(net.parameters(), lr=cfg_data["lr_init"])

    # 4) 准备保存目录
    ckdir = Path(cfg.checkpoint_dir)
    ckdir.mkdir(parents=True, exist_ok=True)

    # 5) 生成文件名并保存
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"muzero_000000000_{ts}.pt"
    target = ckdir / fname

    # checkpoint 内容：net, optim, 原始 cfg_data, game_idx 和空 replay
    ckpt = {
        "net": net.state_dict(),
        "optim": optim.state_dict(),
        "cfg": cfg_data,  # 存字典，不依赖 Config 类
        "game_idx": 0,
        "replay": [],
    }
    torch.save(ckpt, target)
    print("✓ Initialized checkpoint →", target)

    # 6) 更新 latest.pt
    latest = ckdir / "latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(fname)
    except OSError:
        torch.save(ckpt, latest)
    print("✓ Updated latest.pt →", latest)


if __name__ == "__main__":
    gen_init_ckpt("test")

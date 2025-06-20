from pathlib import Path
from datetime import datetime
import torch

from muzero.load_toml_and_pt import load_toml
from muzero.network import MuZeroNet


def gen_init_ckpt(model_name: str):
    model_cfg = load_toml(Path("init_config.toml"))
    # toml 列表 → Python 元组
    model_cfg.obs_shape = tuple(model_cfg.obs_shape)

    # 2) 构造一个简易的 config 对象（无需 import muzero.config）
    ckdir = Path("models") / model_name / "checkpoints"

    # 3) 初始化网络和优化器
    net = MuZeroNet(model_cfg)
    optim = torch.optim.Adam(net.parameters(), lr=model_cfg.lr_init)

    # 4) 准备保存目录
    ckdir.mkdir(parents=True, exist_ok=True)

    # 5) 生成文件名并保存
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"muzero_000000000_{ts}.pt"
    target = ckdir / fname

    ckpt = {
        "net": net.state_dict(),
        "optim": optim.state_dict(),
        "cfg": model_cfg,
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

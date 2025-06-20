# train.py
from muzero.trainer import train
from muzero.init import gen_init_ckpt
from pathlib import Path
from muzero.load_toml_and_pt import load_toml_and_pt
import tomllib

if __name__ == "__main__":
    # 用全量配置跑
    toml_path = Path("train_config.toml")
    data = tomllib.loads(toml_path.read_text())
    model_name = data["model_name"]
    model_path = Path("models") / model_name / "checkpoints/latest.pt"
    if not model_path.exists():
        gen_init_ckpt(model_name)
    cfg = load_toml_and_pt(toml_path, model_path)
    train(cfg)

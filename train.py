# train.py
from muzero.trainer import train
from muzero.init import gen_init_ckpt
from pathlib import Path
from muzero.load_toml_and_pt import load_toml, load_pt_cfg

if __name__ == "__main__":
    toml_path = Path("train_config.toml")
    train_cfg = load_toml(toml_path)
    model_name = train_cfg.model_name
    model_path = Path("models") / model_name / "checkpoints/latest.pt"
    if not model_path.exists():
        gen_init_ckpt(model_name)
    model_cfg = load_pt_cfg(model_path)
    train(train_cfg, model_cfg)

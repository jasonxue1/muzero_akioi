# train.py
from muzero.trainer import train
from muzero.config import Config

if __name__ == "__main__":
    # 用全量配置跑
    train(Config())

# README(中文版)
[🇬🇧 English](README.md)
## 简介
[ak-ioi](apps.ak-ioi.com/oi-2048) 自动化训练, 基于 **muzero** 算法, 兼容 Apple M 系列、Intel CPU 与 CUDA, 支持模型自动初始化与断点续训。
## 快速开始
1. 安装并同步依赖(uv 不需激活虚拟环境)：
   ```bash
   uv sync
   ```
2. 编辑配置：
   * `init_config.toml`(仅首次新建时加载)
   * `train_config.toml`(每次训练时加载, `model_name` 指定模型)
3. 启动训练：
   ```bash
   uv run train.py  # 无需激活虚拟环境
   ```
   或：
   ```bash
   python train.py  # 需先激活虚拟环境
   ```
4. 手动测试：
   ```bash
   uv run test.py  # 无需激活虚拟环境
   ```
   或：
   ```bash
   python test.py  # 需先激活虚拟环境
   ```
## 配置说明
* **init\_config.toml**：初始模型架构, 仅新建时生效。
* **train\_config.toml**：`model_name` 标识模型；若存在则续训, 否则新建并训练。
## 支持环境
* macOS (M 系列/Intel)、Linux + CUDA
## 联系
提交 Issue 或联系维护者。

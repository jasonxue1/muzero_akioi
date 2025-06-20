# README (English)
[🇨🇳 中文版](README.zh-CN.md)
## Overview
[ak-ioi](apps.ak-ioi.com/oi-2048) automated training based on the **muzero** algorithm, compatible with Apple M-series, Intel CPU, and CUDA, supporting automatic initialization and resume training.
## Quick Start
1. Install and sync:
   ```bash
   uv sync
   ```
2. Edit configs:
   * `init_config.toml` (for initial setup)
   * `train_config.toml` (`model_name` for resume or init)
3. Run training:
   ```bash
   uv run train.py  # or python train.py
   ```
## Configuration
* **init\_config.toml**: initial.
* **train\_config.toml**: if `model_name` exists, resume; otherwise init & train.
## Supported
macOS (M-series/Intel), Linux + CUDA
## Contact
Open an Issue or contact the maintainer.

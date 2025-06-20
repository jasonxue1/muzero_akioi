# muzero_akioi
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
3. Start training:

   ```bash
   uv run train.py  # no need to activate the virtual environment
   ```

   or:

   ```bash
   python train.py  # you need to activate the virtual environment first
   ```

4. Manual testing (optional):

   ```bash
   uv run test.py  # no need to activate the virtual environment
   ```

   or:

   ```bash
   python test.py  # you need to activate the virtual environment first
   ```

## Configuration
* **init\_config.toml**: initial.
* **train\_config.toml**: if `model_name` exists, resume; otherwise init & train.
## Supported
macOS (M-series/Intel), Linux + CUDA
## Contact
Open an Issue or contact the maintainer.

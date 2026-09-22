#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h:h}"
cd "$PROJECT_DIR"

uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python "torch==2.7.1" "torchvision==0.22.1" "coremltools==8.3.0"
.venv/bin/python scripts/convert_model.py
xcodegen generate

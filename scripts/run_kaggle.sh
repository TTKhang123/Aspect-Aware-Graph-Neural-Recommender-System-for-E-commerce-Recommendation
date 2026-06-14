#!/usr/bin/env bash
set -e
pip install -q -r requirements.txt
python train_aspect_lightgcn.py --config configs/aspect_lightgcn.yaml

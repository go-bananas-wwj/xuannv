#!/bin/bash
cd /workspace/xuannv
export CUDA_VISIBLE_DEVICES=6,7
exec python3 -u demo_v2/app.py --port 7868

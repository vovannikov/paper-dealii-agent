#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
while pgrep -f baseline_direct.py > /dev/null; do sleep 15; done
python run_extraction.py --prompt v3 --runs runs_v3 --modalities html --cases elasticity stokes --models qwen3-coder:30b devstral-small-2:latest qwen3.6:27b-coding 2>&1 | tee extraction_v3.log
python run_pipeline_stage.py --runs runs_v3 --workers 3
python run_pipeline_stage.py --runs runs_v3 --workers 3 --norm
python recompare.py runs_v3

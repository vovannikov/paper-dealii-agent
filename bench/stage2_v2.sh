#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python run_pipeline_stage.py --runs runs_v2 --workers 3
python run_pipeline_stage.py --runs runs_v2 --workers 3 --norm
python recompare.py runs_v2

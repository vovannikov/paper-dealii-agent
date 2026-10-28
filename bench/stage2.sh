#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python run_pipeline_stage.py "$@" 2>&1 | tee -a pipeline_stage.log

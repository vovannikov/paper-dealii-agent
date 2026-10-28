#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python score_runs.py "$@" 2>&1 | tail -80

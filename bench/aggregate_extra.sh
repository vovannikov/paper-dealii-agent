#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
[ -d runs_v3 ] && python score_runs.py --runs runs_v3 --out scores_v3.json > /dev/null
python aggregate_extra.py "/mnt/c/Users/ivanniko/OneDrive - Hereon/Papers/paper-dealii-agent/tables"

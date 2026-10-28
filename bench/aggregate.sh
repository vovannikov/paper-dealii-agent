#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python score_runs.py > /dev/null
[ -d runs_v2 ] && python score_runs.py --runs runs_v2 --out scores_v2.json > /dev/null
python aggregate.py "/mnt/c/Users/ivanniko/OneDrive - Hereon/Papers/paper-dealii-agent/tables"

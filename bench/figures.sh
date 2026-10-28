#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
mkdir -p "/mnt/c/Users/ivanniko/OneDrive - Hereon/Papers/paper-dealii-agent/figures"
python make_figures.py "/mnt/c/Users/ivanniko/OneDrive - Hereon/Papers/paper-dealii-agent/figures"

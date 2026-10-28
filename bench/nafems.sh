#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python nafems_probe.py 2>&1 | tee nafems.log

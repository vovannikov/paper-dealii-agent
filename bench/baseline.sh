#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
export DEAL_II_DIR=/home/vladimir/development/tpl/dealii-fork-warpack-sundials
grep -h "define DEAL_II_PACKAGE_VERSION" $DEAL_II_DIR/include/deal.II/base/config.h
python baseline_direct.py "$@" 2>&1 | tee -a baseline.log

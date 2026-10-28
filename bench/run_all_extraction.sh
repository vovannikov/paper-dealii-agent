#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python run_extraction.py --models llama3:latest qwen3-coder:30b devstral-small-2:latest qwen3.6:27b-coding laguna-xs-2.1:q4_K_M 2>&1 | tee extraction.log

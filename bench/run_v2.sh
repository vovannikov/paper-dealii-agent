#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python run_extraction.py --prompt v2 --runs runs_v2 --modalities html --models llama3:latest qwen3-coder:30b devstral-small-2:latest qwen3.6:27b-coding 2>&1 | tee extraction_v2.log

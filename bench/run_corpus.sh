#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python corpus.py corpus
ls -la corpus
head -c 1800 corpus/elasticity.html

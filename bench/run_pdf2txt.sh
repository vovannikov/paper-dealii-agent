#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python pdf2txt.py corpus
echo ------ ; cat corpus/stokes.txt | head -60
echo ------ ; head -50 corpus/heat_nafems.txt

#!/bin/bash
set -e
cd /home/vladimir/work/dealii-agent/paper-bench
python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -e "/home/vladimir/development/dealii-agent[llm]" pymupdf latex2mathml numpy matplotlib 2>&1 | tail -5
pip list 2>/dev/null | grep -i -E "femagent|pydantic|openai|numpy|matplotlib|pymupdf|latex2mathml|lxml"

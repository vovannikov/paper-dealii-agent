#!/bin/bash
cd /home/vladimir/work/dealii-agent/paper-bench
source .venv/bin/activate
python - <<'EOF'
import json, glob, collections
for f in sorted(glob.glob("runs/*/*/*/record.json")):
    r = json.load(open(f))
    if r.get("model_ok") and r.get("instance_ok"): continue
    stage = "model" if not r.get("model_ok") else "instance"
    iss = r.get(stage + "_issues") or []
    pe = r.get("model_parse_errors") or []
    calls = r.get("calls", [])
    fin = [c.get("finish") for c in calls]
    msg = (iss[0] if iss else (pe[-1] if pe else r.get(stage+"_error","")))
    print(f.split("/")[1][:12], f.split("/")[2], f.split("/")[3][:10], stage, "attempts", r.get(stage+"_attempts"), fin, "|", str(msg)[:150].replace("\n"," "))
EOF

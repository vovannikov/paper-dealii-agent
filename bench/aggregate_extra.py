"""Extra tables: v3 prompt (elasticity, Stokes) and the direct-generation baseline."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path(sys.argv[1])
LBL = {"qwen3-coder_30b": "Qwen3-Coder 30B", "devstral-small-2_latest": "Devstral Small 2 24B", "qwen3.6_27b-coding": "Qwen3.6 27B (thinking)"}
summary = {}

if (HERE / "scores_v3.json").exists():
    rows = json.loads((HERE / "scores_v3.json").read_text())
    lines = ["\\begin{tabular}{@{}lcccc@{}}\\toprule", "LLM & Elasticity & Stokes & ran (of 2) & exact (of 2) \\\\ \\midrule"]
    for llm, name in LBL.items():
        cells, ran, exact = [], 0, 0
        for case in ("elasticity", "stokes"):
            r = next((x for x in rows if x["llm"] == llm and x["case"] == case), None)
            if r is None:
                cells.append("--"); continue
            run = HERE / "runs_v3" / llm / "html" / case
            ok = False; ex = False
            for n in ("pipeline", "pipeline_norm"):
                p = run / f"{n}.json"
                if p.exists():
                    d = json.loads(p.read_text())
                    if d.get("status") == "ran":
                        ok = True
                        pa = d.get("per_array") or {}
                        ex = bool(pa) and max(v["l2"] for v in pa.values()) < 1e-6
            ran += ok; exact += ex
            valid = r["model_ok"] and r["instance_ok"]
            cells.append("\\cmark" if r["semantic_ok"] else ("$\\sim$" if valid else "\\xmark"))
            summary[f"{llm}|{case}"] = dict(correct=r["semantic_ok"], valid=bool(valid), ran=ok, exact=ex, mod=r.get("eq_modulo"))
        lines.append(f"{name} & {cells[0]} & {cells[1]} & {ran} & {exact} \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    (OUT / "tab_v3.tex").write_text("\n".join(lines))

base = []
for p in sorted((HERE / "baseline_runs").glob("*/baseline.json")):
    base.append(json.loads(p.read_text()))
if base:
    lines = ["\\begin{tabular}{@{}lccrrr@{}}\\toprule", "LLM & attempts & outcome & probe error & tokens (in+out) & wall time \\\\ \\midrule"]
    for b in base:
        last = b["attempts"][-1]
        stage = last["stage"]
        out = {"ok": "ran", "compile": "no compile", "run": "run failure", "output": "no output", "configure": "configure error", "llm_error": "no compile (truncated)"}[stage]
        err = f"{100 * last['rel_error']:.2g}\\%" if stage == "ok" else "--"
        name = LBL.get(b["model"].replace(":", "_"), b["model"])
        tok = f"{b['prompt_tokens']:,}+{b['completion_tokens']:,}".replace(",", "\\,")
        lines.append(f"{name} & {len(b['attempts'])} & {out} & {err} & {tok} & {b['seconds']:.0f}\\,s \\\\")
        summary["baseline|" + b["model"]] = dict(attempts=[(a["stage"], a.get("rel_error")) for a in b["attempts"]], pt=b["prompt_tokens"], ct=b["completion_tokens"], s=b["seconds"])
    lines.append("\\bottomrule\\end{tabular}")
    (OUT / "tab_baseline.tex").write_text("\n".join(lines))
print(json.dumps(summary, indent=1, default=str))

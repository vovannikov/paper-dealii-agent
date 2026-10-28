"""Aggregate scores + pipeline results into LaTeX tables and macros for the paper."""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "tables"
OUT.mkdir(parents=True, exist_ok=True)

CASES = ["heat_nafems", "poisson", "transient_heat", "elasticity", "stokes", "allen_cahn"]
CASE_LABEL = {"heat_nafems": "Heat (NAFEMS)", "poisson": "Poisson", "transient_heat": "Transient heat",
              "elasticity": "Elasticity", "stokes": "Stokes", "allen_cahn": "Allen--Cahn"}
LLMS = ["llama3_latest", "qwen3-coder_30b", "devstral-small-2_latest", "qwen3.6_27b-coding", "laguna-xs-2.1_q4_K_M"]
LLM_LABEL = {"llama3_latest": "Llama 3 8B", "qwen3-coder_30b": "Qwen3-Coder 30B", "devstral-small-2_latest": "Devstral Small 2 24B",
             "qwen3.6_27b-coding": "Qwen3.6 27B (thinking)", "laguna-xs-2.1_q4_K_M": "Laguna XS 2.1 33B (thinking)"}


def load(scores_file: str, runs_dir: str):
    rows = json.loads((HERE / scores_file).read_text())
    for r in rows:
        run = HERE / runs_dir / r["llm"] / r["modality"] / r["case"]
        r["valid"] = bool(r["model_ok"] and r["instance_ok"])
        for name in ("pipeline", "pipeline_norm"):
            p = run / f"{name}.json"
            r[name] = json.loads(p.read_text()) if p.exists() else None
        strict = r["pipeline"] and r["pipeline"].get("status") == "ran"
        norm = r["pipeline_norm"] and r["pipeline_norm"].get("status") == "ran"
        r["ran_strict"] = bool(strict)
        r["ran"] = bool(strict or norm)
        res = r["pipeline"] if strict else (r["pipeline_norm"] if norm else None)
        pa = (res or {}).get("per_array") or {}
        r["sol_l2"] = max((v["l2"] for v in pa.values()), default=None)
        r["e2e_exact"] = r["ran"] and r["sol_l2"] is not None and r["sol_l2"] < 1e-6
        rec = json.loads((run / "record.json").read_text())
        r["record"] = rec
        # failure category
        cat = None
        if not r["model_ok"]:
            msgs = " ".join(rec.get("model_issues") or []) + " ".join(rec.get("model_parse_errors") or [])
            cat = ("truncated" if "truncated" in msgs else "dimension" if "dimension_mismatch" in msgs else
                   "rank" if "rank_mismatch" in msgs else "schema")
        elif not r["instance_ok"]:
            msgs = " ".join(rec.get("instance_issues") or []) + rec.get("instance_error", "")
            cat = "truncated" if "truncated" in msgs else "missing value" if "missing_parameter_value" in msgs else "instance"
        elif not r["semantic_ok"]:
            reasons = " ".join(r["reasons"])
            cat = ("equation" if "equation" in reasons else "initial data" if "initial condition" in reasons else
                   "boundary data" if "boundary data" in reasons else "geometry")
        r["category"] = cat
        # time coordinate declared as a parameter
        mj = run / "model.json"
        r["t_param"] = False
        if mj.exists() and r["case"] in ("transient_heat",):
            d = json.loads(mj.read_text())
            r["t_param"] = any(p["name"].lower() in ("t", "time") for p in d.get("parameters", []))
    return rows


def count(rows, **kw):
    return sum(all(r.get(k) == v for k, v in kw.items()) for r in rows)


def summarize(rows):
    s = {}
    for llm in LLMS:
        for mod in ("html", "pdf"):
            sub = [r for r in rows if r["llm"] == llm and r["modality"] == mod]
            if not sub:
                continue
            s[(llm, mod)] = dict(n=len(sub), valid=sum(r["valid"] for r in sub), correct=sum(r["semantic_ok"] for r in sub),
                                 ran=sum(r["ran"] for r in sub), exact=sum(r["e2e_exact"] for r in sub),
                                 first=sum(r["first_try"] and r["valid"] for r in sub),
                                 silent=sum(r["valid"] and not r["semantic_ok"] and r["ran"] for r in sub),
                                 caught=sum(r["valid"] and not r["semantic_ok"] and not r["ran"] for r in sub))
    return s


def cell(sym):
    return {"ok": "\\cmark", "wrong": "$\\sim$", "bad": "\\xmark"}[sym]


def sym(r):
    return "ok" if r["semantic_ok"] else ("wrong" if r["valid"] else "bad")


def main():
    v1 = load("scores.json", "runs")
    S = summarize(v1)
    lines = []
    # Table: summary per LLM
    lines.append("\\begin{tabular}{@{}lcccccccc@{}}\\toprule")
    lines.append("& \\multicolumn{4}{c}{HTML (MathML)} & \\multicolumn{4}{c}{PDF text} \\\\ \\cmidrule(lr){2-5}\\cmidrule(lr){6-9}")
    lines.append("LLM & valid & correct & ran & exact & valid & correct & ran & exact \\\\ \\midrule")
    tot = Counter()
    for llm in LLMS:
        h, p = S[(llm, "html")], S[(llm, "pdf")]
        lines.append(f"{LLM_LABEL[llm]} & {h['valid']} & {h['correct']} & {h['ran']} & {h['exact']} & {p['valid']} & {p['correct']} & {p['ran']} & {p['exact']} \\\\")
        for k in ("valid", "correct", "ran", "exact"):
            tot[("html", k)] += h[k]
            tot[("pdf", k)] += p[k]
    lines.append("\\midrule")
    lines.append("Total (of 30 each) & " + " & ".join(str(tot[(m, k)]) for m in ("html", "pdf") for k in ("valid", "correct", "ran", "exact")) + " \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    (OUT / "tab_summary.tex").write_text("\n".join(lines))

    # Table: matrix
    lines = ["\\begin{tabular}{@{}l" + "cc" * len(LLMS) + "@{}}\\toprule"]
    lines.append("Document & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{LLM_LABEL[l].split(' (')[0]}}}" for l in LLMS) + " \\\\")
    lines.append(" & " + " & ".join("H & P" for _ in LLMS) + " \\\\ \\midrule")
    for c in CASES:
        row = [CASE_LABEL[c]]
        for l in LLMS:
            for m in ("html", "pdf"):
                r = next(x for x in v1 if x["llm"] == l and x["modality"] == m and x["case"] == c)
                row.append(cell(sym(r)))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    (OUT / "tab_matrix.tex").write_text("\n".join(lines))

    # Table: cost
    lines = ["\\begin{tabular}{@{}lrrrrrr@{}}\\toprule",
             "LLM & calls & prompt tok. & output tok. & reasoning share & median s/run & max s/run \\\\ \\midrule"]
    cost = {}
    for llm in LLMS:
        sub = [r for r in v1 if r["llm"] == llm]
        calls = statistics.mean(r["llm_calls"] for r in sub)
        pt = statistics.mean(r["prompt_tokens"] for r in sub)
        ct = statistics.mean(r["completion_tokens"] for r in sub)
        secs = [r["seconds"] for r in sub if r["seconds"]]
        rs = sum(r["reasoning_chars"] for r in sub)
        ans = sum(sum(c.get("answer_chars", 0) for c in r["record"].get("calls", [])) for r in sub)
        share = rs / (rs + ans) if rs + ans else 0
        cost[llm] = dict(calls=calls, pt=pt, ct=ct, med=statistics.median(secs), mx=max(secs), share=share)
        lines.append(f"{LLM_LABEL[llm]} & {calls:.1f} & {pt:,.0f} & {ct:,.0f} & {100*share:.0f}\\% & {statistics.median(secs):.0f} & {max(secs):.0f} \\\\".replace(",", "\\,"))
    lines.append("\\bottomrule\\end{tabular}")
    (OUT / "tab_cost.tex").write_text("\n".join(lines))

    # Failure taxonomy
    cats = Counter(r["category"] for r in v1 if r["category"])
    order = ["schema", "rank", "dimension", "truncated", "missing value", "instance", "equation", "initial data", "boundary data", "geometry"]
    names = {"schema": "Schema/codec error (missing or mistyped key)", "rank": "Rank mismatch (validator)",
             "dimension": "Dimension mismatch (validator)", "truncated": "Output truncated (runaway reasoning)",
             "missing value": "Parameter value missing (validator)", "instance": "Instance not valid (other)",
             "equation": "Equation not equivalent to gold", "initial data": "Initial condition wrong or missing",
             "boundary data": "Boundary data wrong (value, type or side)", "geometry": "Domain box wrong"}
    lines = ["\\begin{tabular}{@{}lr@{}}\\toprule", "Outcome of a non-correct run & runs \\\\ \\midrule"]
    lines.append("\\multicolumn{2}{@{}l}{\\emph{Rejected at the trust boundary}} \\\\")
    for c in order[:6]:
        if cats[c]:
            lines.append(f"\\quad {names[c]} & {cats[c]} \\\\")
    lines.append("\\multicolumn{2}{@{}l}{\\emph{Accepted by the validators, but not equivalent to the paper}} \\\\")
    for c in order[6:]:
        if cats[c]:
            lines.append(f"\\quad {names[c]} & {cats[c]} \\\\")
    lines.append("\\bottomrule\\end{tabular}")
    (OUT / "tab_failures.tex").write_text("\n".join(lines))

    # v2
    summary = {"v1": {f"{k[0]}|{k[1]}": v for k, v in S.items()}, "cost": cost, "cats": dict(cats)}
    if (HERE / "scores_v2.json").exists():
        v2 = load("scores_v2.json", "runs_v2")
        lines = ["\\begin{tabular}{@{}lcccccc@{}}\\toprule",
                 "& \\multicolumn{3}{c}{original prompt} & \\multicolumn{3}{c}{extended prompt} \\\\ \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
                 "LLM & valid & correct & ran & valid & correct & ran \\\\ \\midrule"]
        for llm in LLMS:
            a = [r for r in v1 if r["llm"] == llm and r["modality"] == "html"]
            b = [r for r in v2 if r["llm"] == llm]
            if not b:
                continue
            lines.append(f"{LLM_LABEL[llm].split(' (')[0]} & {sum(r['valid'] for r in a)} & {sum(r['semantic_ok'] for r in a)} & {sum(r['ran'] for r in a)} & "
                         f"{sum(r['valid'] for r in b)}/{len(b)} & {sum(r['semantic_ok'] for r in b)}/{len(b)} & {sum(r['ran'] for r in b)}/{len(b)} \\\\")
        lines.append("\\bottomrule\\end{tabular}")
        (OUT / "tab_v2.tex").write_text("\n".join(lines))
        lines = ["\\begin{tabular}{@{}l" + "c" * len(CASES) + "@{}}\\toprule", "LLM & " + " & ".join(CASE_LABEL[c] for c in CASES) + " \\\\ \\midrule"]
        for llm in LLMS:
            b = [r for r in v2 if r["llm"] == llm]
            if not b:
                continue
            row = [LLM_LABEL[llm].split(" (")[0]]
            for c in CASES:
                rr = next((x for x in b if x["case"] == c), None)
                row.append(cell(sym(rr)) if rr else "--")
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\bottomrule\\end{tabular}")
        (OUT / "tab_v2_matrix.tex").write_text("\n".join(lines))
        summary["v2"] = {l: dict(valid=sum(r["valid"] for r in v2 if r["llm"] == l), correct=sum(r["semantic_ok"] for r in v2 if r["llm"] == l),
                                  ran=sum(r["ran"] for r in v2 if r["llm"] == l), n=count(v2, llm=l)) for l in LLMS}
        summary["v2_cats"] = dict(Counter(r["category"] for r in v2 if r["category"]))
    # extra facts
    summary["t_param"] = sum(r["t_param"] for r in v1)
    summary["silent_total"] = sum(r["valid"] and not r["semantic_ok"] and r["ran"] for r in v1)
    summary["valid_wrong_total"] = sum(r["valid"] and not r["semantic_ok"] for r in v1)
    summary["valid_total"] = sum(r["valid"] for r in v1)
    summary["correct_total"] = sum(r["semantic_ok"] for r in v1)
    summary["ran_correct"] = sum(r["semantic_ok"] and r["ran"] for r in v1)
    summary["exact_total"] = sum(r["e2e_exact"] for r in v1)
    summary["first_try_valid"] = sum(r["first_try"] and r["valid"] for r in v1)
    summary["repaired"] = sum(r["valid"] and not r["first_try"] for r in v1)
    summary["correct_after_repair"] = sum(r["semantic_ok"] and not r["first_try"] for r in v1)
    summary["modulo"] = sum(1 for r in v1 if r["semantic_ok"] and r.get("eq_modulo"))
    summary["ran_wrong_list"] = [f"{r['llm']}|{r['modality']}|{r['case']}|{r['category']}|{r['sol_l2']}" for r in v1 if r["valid"] and not r["semantic_ok"] and r["ran"]]
    summary["rejected_by_generator"] = [f"{r['llm']}|{r['modality']}|{r['case']}" for r in v1 if r["semantic_ok"] and not r["ran"]]
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()

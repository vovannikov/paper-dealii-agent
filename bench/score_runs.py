"""Score every run directory against the gold model; `--selftest` checks the scorer on gold-vs-gold and mutants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/vladimir/development/dealii-agent")

from femagent.instance import ProblemInstance  # noqa: E402
from femagent.ir.expr import Constant, Parameter  # noqa: E402
from femagent.ir.model import PDEModel  # noqa: E402

from gold import GOLD  # noqa: E402
from score import score  # noqa: E402

HERE = Path(__file__).parent


def score_run(run: Path, case: str) -> dict:
    g = GOLD[case]
    record = json.loads((run / "record.json").read_text())
    model = PDEModel.model_validate_json((run / "model.json").read_text()) if (run / "model.json").exists() else None
    inst = ProblemInstance.model_validate_json((run / "instance.json").read_text()) if (run / "instance.json").exists() else None
    s = score(g["model"], g["instance"], model, inst, incompressible=g.get("incompressible", False),
              free_scale=g.get("free_scale", ()))
    calls = record.get("calls", [])
    s.update(
        model_ok=record.get("model_ok", False), instance_ok=record.get("instance_ok", False),
        model_attempts=record.get("model_attempts"), instance_attempts=record.get("instance_attempts"),
        first_try=bool(record.get("model_first_ok") and record.get("instance_first_ok")),
        seconds=record.get("total_seconds"), llm_calls=len(calls),
        prompt_tokens=sum(c.get("prompt_tokens") or 0 for c in calls),
        completion_tokens=sum(c.get("completion_tokens") or 0 for c in calls),
        reasoning_chars=sum(c.get("reasoning_chars") or 0 for c in calls),
        document_chars=record.get("document_chars"),
    )
    return s


def selftest() -> None:
    for case, g in GOLD.items():
        s = score(g["model"], g["instance"], g["model"], g["instance"], incompressible=g.get("incompressible", False),
                  free_scale=g.get("free_scale", ()))
        print(f"gold-vs-gold {case:15s} semantic_ok={s['semantic_ok']} eq={s.get('eq_ok')} bc={s.get('bc_sides_ok')}/{s.get('bc_sides_total')} {s['reasons']}")
    # mutants of the heat case: flipped convective sign, wrong value, wrong side map
    g = GOLD["heat_nafems"]
    bad = g["model"].model_copy(deep=True)
    for bc in bad.boundary_conditions:
        if bc.boundary_id == "top":
            bc.value = Parameter("h") * (g["model"].fields[0]) - Parameter("T_amb")
    print("mutant flipped robin :", score(g["model"], g["instance"], bad, g["instance"])["reasons"])
    wrong = g["instance"].model_copy(update={"parameter_values": {**g["instance"].parameter_values, "k": 5.2}})
    print("mutant wrong k       :", score(g["model"], g["instance"], g["model"], wrong)["reasons"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--runs", default=str(HERE / "runs"))
    parser.add_argument("--out", default=str(HERE / "scores.json"))
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    rows = []
    for record in sorted(Path(args.runs).glob("*/*/*/record.json")):
        run = record.parent
        model_tag, modality, case = run.parent.parent.name, run.parent.name, run.name
        if case not in GOLD:
            continue
        row = score_run(run, case)
        row.update(llm=model_tag, modality=modality, case=case)
        rows.append(row)
    Path(args.out).write_text(json.dumps(rows, indent=1, default=str))
    print(f"{len(rows)} runs scored -> {args.out}")
    for r in rows:
        print(f"{r['llm']:28s} {r['modality']:4s} {r['case']:15s} valid={r['model_ok'] and r['instance_ok']!s:5} "
              f"semantic={r['semantic_ok']!s:5} eq={r.get('eq_ok')} bc={r.get('bc_sides_ok')}/{r.get('bc_sides_total')} "
              f"{'; '.join(r['reasons'])[:110]}")


if __name__ == "__main__":
    main()

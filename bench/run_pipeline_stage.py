"""Stage 2: generate, build and run the solver for every extraction (verification gate off) and compare its solution
with the solution of the gold model on the same mesh.

    python run_pipeline_stage.py --gold                # gold solutions first
    python run_pipeline_stage.py --workers 4           # then every run directory under runs/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/vladimir/development/dealii-agent")

from femagent.instance import ProblemInstance  # noqa: E402
from femagent.ir.model import PDEModel  # noqa: E402
from femagent.ir.validate import has_time_derivative  # noqa: E402
from femagent.pipeline import run_pipeline  # noqa: E402
from femagent.solverplan import plan_discretization  # noqa: E402
from femagent.verification.vtu import read_vtu  # noqa: E402

from gold import GOLD  # noqa: E402

HERE = Path(__file__).parent
DEAL_II = "/home/vladimir/development/tpl/dealii-fork-warpack-sundials"


def _solve(model: PDEModel, instance: ProblemInstance, case: str, work: Path, verify: bool = False):
    time_opts = GOLD[case]["time"]
    transient = has_time_derivative(model)
    dt = time_opts["dt"] if (time_opts and transient) else None
    steps = time_opts["num_steps"] if (time_opts and transient) else None
    plan = plan_discretization(model, dt=dt)
    return run_pipeline(model, instance, deal_ii_dir=DEAL_II, work_dir=work, plan=plan, num_steps=steps, dt=dt,
                        verify=verify, describe=False)


def _collect(report, work: Path, keep: Path) -> dict:
    stages = {s.name: s.status for s in report.stages}
    details = {s.name: s.detail for s in report.stages}
    issues = [str(i) for s in report.stages for i in s.issues if s.status == "failed"][:6]
    vtus = sorted(work.glob("*/build/*.vtu"))
    solution = None
    if vtus:
        keep.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(vtus[-1], keep)
        solution = str(keep)
    for build in work.glob("*/build"):
        shutil.rmtree(build, ignore_errors=True)
    return dict(stages=stages, details=details, issues=issues, solution=solution, passed=report.succeeded)


def _compare(sol: Path, ref: Path) -> dict:
    a, b = read_vtu(sol), read_vtu(ref)
    arrays_a, arrays_b = list(a.arrays.values()), list(b.arrays.values())
    if len(a.points) != len(b.points) or len(arrays_a) != len(arrays_b):
        return dict(comparable=False, reason=f"mesh/fields differ ({len(a.points)} vs {len(b.points)} points)")
    worst = 0.0
    for x, y in zip(arrays_a, arrays_b):
        if len(x) != len(y):
            return dict(comparable=False, reason="field layout differs")
        scale = max(abs(v) for v in y) or 1.0
        worst = max(worst, max(abs(p - q) for p, q in zip(x, y)) / scale)
    return dict(comparable=True, rel_linf=worst)


def run_gold(case: str) -> dict:
    g = GOLD[case]
    out = HERE / "gold_runs" / case
    shutil.rmtree(out, ignore_errors=True)
    start = time.time()
    report = _solve(g["model"], g["instance"], case, out / "work")
    result = _collect(report, out / "work", out / "solution.vtu")
    result["seconds"] = time.time() - start
    (out / "pipeline.json").write_text(json.dumps(result, indent=1))
    return result


def run_extracted(run: Path, case: str, norm: bool = False) -> dict:
    g = GOLD[case]
    result: dict = {}
    out_name = "pipeline_norm.json" if norm else "pipeline.json"
    if not ((run / "model.json").exists() and (run / "instance.json").exists()):
        result = dict(status="no_extraction")
    else:
        text = (run / "model.json").read_text()
        if norm:
            import re

            text = re.sub(r'"space": "(L2|Hdiv|H\(div\)|Hcurl|H\(curl\))"', '"space": "H1"', text)
        model = PDEModel.model_validate_json(text)
        inst = ProblemInstance.model_validate_json((run / "instance.json").read_text())
        geom = inst.geometry.model_copy(update={"refinements": g["instance"].geometry.refinements}) if inst.geometry else None
        inst = inst.model_copy(update=dict(geometry=geom, field_degrees=g["instance"].field_degrees,
                                           pin_fields=g["instance"].pin_fields))
        start = time.time()
        try:
            shutil.rmtree(run / "work", ignore_errors=True)
            report = _solve(model, inst, case, run / "work")
            result = _collect(report, run / "work", run / "solution.vtu")
            result["status"] = "ran" if report.succeeded else "failed"
            ref = HERE / "gold_runs" / case / "solution.vtu"
            if result.get("solution") and ref.exists():
                result["compare"] = _compare(Path(result["solution"]), ref)
        except Exception as exc:  # generator crash etc.
            result = dict(status="crashed", error=f"{type(exc).__name__}: {str(exc)[:300]}")
        result["seconds"] = time.time() - start
    (run / out_name).write_text(json.dumps(result, indent=1))
    return result


def _job(args):
    run, case, norm = args
    return str(run), run_extracted(Path(run), case, norm)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--runs", default=str(HERE / "runs"))
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--norm", action="store_true", help="only runs with a non-H1 space annotation, normalised to H1")
    args = parser.parse_args()
    if args.gold:
        for case in GOLD:
            r = run_gold(case)
            print(f"gold {case}: {r['stages']} {r['seconds']:.0f}s", flush=True)
        return
    jobs = []
    for record in sorted(Path(args.runs).glob("*/*/*/record.json")):
        run = record.parent
        if run.name not in GOLD:
            continue
        if args.norm:
            mj = run / "model.json"
            if mj.exists() and (run / "instance.json").exists() and any(
                f'"space": "{s}"' in mj.read_text() for s in ("L2", "Hdiv", "Hcurl", "H(div)", "H(curl)")
            ):
                jobs.append((str(run), run.name, True))
        elif args.redo or not (run / "pipeline.json").exists():
            jobs.append((str(run), run.name, False))
    print(f"{len(jobs)} runs to execute", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        for run, result in pool.map(_job, jobs):
            print(f"{Path(run).relative_to(args.runs)}: {result.get('status')} {result.get('compare', '')}", flush=True)


if __name__ == "__main__":
    main()

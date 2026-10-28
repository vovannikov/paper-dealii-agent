"""NAFEMS probe: temperature at P = (0.6, 0.2) of the heat benchmark on a sequence of meshes (reference: 18.25 degC).

The generated solver is run on the gold model for refinements r = 3..9 (2^r x 2^r cells, Q1 and Q2). P lies on the right
edge, where a Q1 trace is exactly linear between the vertices, so linear interpolation along the edge is the finite-element
value for Q1; for Q2 the vertex values are interpolated linearly (an O(h^2) post-processing error).
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/vladimir/development/dealii-agent")

from femagent.instance import ProblemInstance  # noqa: E402
from femagent.pipeline import run_pipeline  # noqa: E402
from femagent.solverplan import plan_discretization  # noqa: E402
from femagent.verification.vtu import read_vtu  # noqa: E402

from gold import GOLD  # noqa: E402

HERE = Path(__file__).parent
DEAL_II = "/home/vladimir/development/tpl/dealii-fork-warpack-sundials"
T_REF = 18.25


def probe(vtu: Path, x0=0.6, y0=0.2) -> float:
    data = read_vtu(vtu)
    values = next(iter(data.arrays.values()))
    edge = sorted((p[1], v) for p, v in zip(data.points, values) if abs(p[0] - x0) < 1e-5)
    for (ya, va), (yb, vb) in zip(edge, edge[1:]):
        if ya - 1e-6 <= y0 <= yb + 1e-6:
            return va + (vb - va) * (y0 - ya) / (yb - ya)
    raise ValueError("probe point not found")


def job(arg):
    refinement, degree = arg
    g = GOLD["heat_nafems"]
    inst: ProblemInstance = g["instance"].model_copy(update=dict(
        geometry=g["instance"].geometry.model_copy(update={"refinements": refinement}),
        field_degrees={"T": degree} if degree > 1 else {},
    ))
    work = HERE / "nafems" / f"r{refinement}_q{degree}"
    report = run_pipeline(g["model"], inst, deal_ii_dir=DEAL_II, work_dir=work, plan=plan_discretization(g["model"]),
                          verify=False, describe=False)
    vtu = sorted(work.glob("*/build/*.vtu"))
    value = probe(vtu[-1]) if vtu and report.succeeded else None
    detail = {s.name: s.detail for s in report.stages}.get("run", "")
    return dict(refinement=refinement, degree=degree, value=value, run=detail)


if __name__ == "__main__":
    jobs = [(r, d) for d in (1, 2) for r in range(3, 10)]
    rows = []
    with ProcessPoolExecutor(3) as pool:
        for row in pool.map(job, jobs):
            row["rel_error"] = None if row["value"] is None else abs(row["value"] - T_REF) / T_REF
            rows.append(row)
            print(row, flush=True)
    (HERE / "nafems_results.json").write_text(json.dumps(rows, indent=1))

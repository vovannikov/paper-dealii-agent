"""Add per-array relative differences to every pipeline result (no rebuild)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/vladimir/development/dealii-agent")
from femagent.verification.vtu import read_vtu  # noqa: E402

HERE = Path(__file__).parent


def per_array(sol, ref):
    a, b = read_vtu(sol), read_vtu(ref)
    if len(a.points) != len(b.points) or len(a.arrays) != len(b.arrays):
        return None
    out = {}
    for (na, x), (nb, y) in zip(a.arrays.items(), b.arrays.items()):
        if len(x) != len(y):
            return None
        scale = max(abs(v) for v in y) or 1.0
        nrm = sum(q * q for q in y) ** 0.5 or 1.0
        out[nb] = dict(linf=max(abs(p - q) for p, q in zip(x, y)) / scale,
                       l2=sum((p - q) ** 2 for p, q in zip(x, y)) ** 0.5 / nrm)
    return out


for root in sys.argv[1:]:
    for pj in sorted(Path(root).glob("*/*/*/pipeline*.json")):
        d = json.loads(pj.read_text())
        sol = d.get("solution")
        ref = HERE / "gold_runs" / pj.parent.name / "solution.vtu"
        if sol and ref.exists():
            d["per_array"] = per_array(sol, ref)
            pj.write_text(json.dumps(d, indent=1))
            print(pj.relative_to(root), d["per_array"])

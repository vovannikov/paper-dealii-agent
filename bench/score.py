"""Semantic scoring of an extracted (model, instance) against the gold one.

Names chosen by the LLM are irrelevant: operators and boundary data are compared by *evaluating* them. Both models'
equations are evaluated on the same random polynomial test fields (each model with its own parameter values) and the
residual polynomials are compared up to a global factor. Boundary data are compared side by side of the rectangle
(Dirichlet vs flux class and value as a polynomial of the test fields), so a wrong convective sign, a wrong side or a
missing condition is detected. Incompressible problems are also compared on divergence-free fields, so that
-div(2 nu eps(u)) and -nu laplacian(u) are recognised as equal modulo the constraint (reported separately).
"""
from __future__ import annotations

import itertools
import random
import sys

sys.path.insert(0, "/home/vladimir/development/dealii-agent")

import femagent.verification.manufactured as M  # noqa: E402
from femagent.ir.model import BCType, PDEModel  # noqa: E402
from femagent.ir.spaces import Rank  # noqa: E402
from femagent.instance import ProblemInstance  # noqa: E402

DIM = 2
TOL = 1e-7


def _rand_poly(rng: random.Random, degree: int = 3) -> dict:
    poly = {}
    for ex in range(degree + 1):
        for ey in range(degree + 1 - ex):
            for et in (0, 1):
                if rng.random() < 0.55:
                    poly[(ex, ey, et)] = rng.choice([-3, -2, -1, 1, 2, 3]) / 2.0
    return poly or {(1, 0, 0): 1.0}


def _test_fields(model: PDEModel, rng: random.Random, divfree: bool) -> dict:
    out = {}
    for f in model.fields:
        if f.rank is Rank.VECTOR:
            if divfree:
                psi = _rand_poly(rng, 4)
                out[f.name] = (M._poly_diff(psi, 1), M._poly_scale(M._poly_diff(psi, 0), -1.0))
            else:
                out[f.name] = (_rand_poly(rng), _rand_poly(rng))
        elif f.rank is Rank.SCALAR:
            out[f.name] = _rand_poly(rng)
        else:
            raise NotImplementedError("tensor fields are not supported by the scorer")
    return out


def _flat(r, prefix=()) -> dict:
    if isinstance(r, dict):
        return {(prefix, e): c for e, c in r.items() if abs(c) > 1e-12}
    out = {}
    for i, comp in enumerate(r):
        out.update(_flat(comp, prefix + (i,)))
    return out


def _close(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(abs(a[k] - b[k]) <= TOL * max(1.0, abs(b[k])) for k in a)


def _proportional(a: dict, b: dict, unit: bool = True) -> bool:
    """`b = ratio * a`; with `unit` the ratio must be +-1 (boundary fluxes are defined against the equation as written)."""
    if not a or not b or a.keys() != b.keys():
        return False
    k0 = max(a, key=lambda k: abs(a[k]))
    ratio = b[k0] / a[k0]
    if ratio == 0 or (unit and abs(abs(ratio) - 1.0) > 1e-7):
        return False
    return all(abs(b[k] - ratio * a[k]) <= TOL * max(1.0, abs(b[k])) for k in a)


def _relations(model: PDEModel) -> dict:
    return {r.name: r.definition for r in model.constitutive_relations}


def _residuals(model, instance, exact) -> list:
    out = []
    for eq in model.equations:
        try:
            out.append(_flat(M._residual_poly(eq, exact, instance.parameter_values, _relations(model), DIM)))
        except Exception as exc:  # unevaluable: missing value, unsupported node, rank clash, ...
            out.append(f"error: {type(exc).__name__}: {str(exc)[:100]}")
    return out


def _mappings(gold: PDEModel, ext: PDEModel):
    """Candidate gold-field-name -> extracted-field-name bijections (name based first, then by rank)."""
    by_name = {f.name.lower(): f for f in ext.fields}
    named = {g.name: by_name[g.name.lower()].name for g in gold.fields
             if g.name.lower() in by_name and by_name[g.name.lower()].rank == g.rank}
    seen = []
    if len(named) == len(gold.fields):
        seen.append(named)
    groups = []
    for rank in (Rank.SCALAR, Rank.VECTOR):
        g = [f.name for f in gold.fields if f.rank is rank]
        e = [f.name for f in ext.fields if f.rank is rank]
        if len(g) != len(e):
            continue
        groups.append([list(zip(g, p)) for p in itertools.permutations(e)])
    for combo in itertools.product(*groups) if groups else []:
        mapping = {g: e for pairs in combo for g, e in pairs}
        if len(mapping) == len(gold.fields) and mapping not in seen:
            seen.append(mapping)
    if not seen:
        seen.append(named)
    return seen


def _side_table(model, instance, exact, rename) -> dict | None:
    if instance.geometry is None:
        return None
    table = {}
    geometry = instance.geometry
    for bc in model.boundary_conditions:
        if bc.boundary_id not in geometry.boundary_ids or bc.field not in rename:
            continue
        try:
            value = _flat(M._evaluate(bc.value, exact, instance.parameter_values, _relations(model), DIM))
        except Exception as exc:
            value = f"error: {type(exc).__name__}"
        cls = "dirichlet" if bc.type is BCType.DIRICHLET else "flux"
        for side in geometry.sides(bc.boundary_id):
            table[(rename[bc.field], side)] = (cls, value)
    return table


def _score_mapping(gold, ginst, ext, einst, mapping, incompressible: bool, seed: int, free_scale=()) -> dict:
    rng = random.Random(seed)
    res = {"mapping": mapping, "eq": [], "reasons": []}
    families = [False] + ([True] if incompressible else [])
    fam_data = []
    for divfree in families:
        g_exact = _test_fields(gold, rng, divfree)
        e_exact = {mapping[g]: g_exact[g] for g in mapping}
        fam_data.append((divfree, g_exact, e_exact, _residuals(gold, ginst, g_exact), _residuals(ext, einst, e_exact)))

    n_matched = 0
    for j, eq in enumerate(gold.equations):
        outcome = "unmatched"
        for divfree, _, _, gres, eres in fam_data:
            g = gres[j]
            if isinstance(g, str) or not g:
                continue
            if any(isinstance(e, dict) and _proportional(g, e, unit=j not in free_scale) for e in eres):
                outcome = "modulo_divfree" if divfree else "exact"
                if not divfree:
                    break
        if outcome == "unmatched":
            errors = [e for e in fam_data[0][4] if isinstance(e, str)]
            res["reasons"].append(f"equation {j} not reproduced" + (f" ({errors[0]})" if errors else ""))
        else:
            n_matched += 1
        res["eq"].append(outcome)
    res["n_matched"] = n_matched
    res["eq_ok"] = n_matched == len(gold.equations)
    res["eq_modulo"] = "modulo_divfree" in res["eq"]
    res["n_extra_eq"] = max(0, len(ext.equations) - n_matched)

    # initial conditions (value polynomials of x, y)
    ic_ok = True
    for ic in gold.initial_conditions:
        match = [e for e in ext.initial_conditions if e.field == mapping.get(ic.field)]
        try:
            gv = _flat(M._evaluate(ic.value, {}, ginst.parameter_values, _relations(gold), DIM))
            ev = _flat(M._evaluate(match[0].value, {}, einst.parameter_values, _relations(ext), DIM)) if match else None
        except Exception:
            ev = None
        if ev is None or not _close(gv, ev):
            ic_ok = False
            res["reasons"].append(f"initial condition of '{ic.field}' wrong or missing")
    res["ic_ok"] = ic_ok

    # boundary data side by side
    g_exact, e_exact = fam_data[0][1], fam_data[0][2]
    inv = {e: g for g, e in mapping.items()}
    gt = _side_table(gold, ginst, g_exact, {f.name: f.name for f in gold.fields})
    et = _side_table(ext, einst, e_exact, inv)
    bc_total = bc_ok = 0
    if et is None:
        res["reasons"].append("no geometry")
    else:
        for f in gold.fields:
            for side in range(2 * DIM):
                gcls, gval = gt.get((f.name, side), ("flux", {}))
                ecls, eval_ = et.get((f.name, side), ("flux", {}))
                bc_total += 1
                same = gcls == ecls and isinstance(gval, dict) and isinstance(eval_, dict) and _close(gval, eval_)
                bc_ok += same
                if not same:
                    res["reasons"].append(f"boundary data of '{f.name}' on side {side} differ ({gcls} vs {ecls})")
    res["bc_sides_ok"], res["bc_sides_total"] = bc_ok, bc_total
    res["bc_ok"] = et is not None and bc_ok == bc_total
    return res


def score(gold: PDEModel, ginst: ProblemInstance, ext: PDEModel | None, einst: ProblemInstance | None,
          incompressible: bool = False, seed: int = 7, free_scale=()) -> dict:
    out = {"semantic_ok": False, "reasons": []}
    if ext is None:
        out["reasons"].append("no model")
        return out
    gnames = {(f.name.lower(), f.rank.value) for f in gold.fields}
    enames = {(f.name.lower(), f.rank.value) for f in ext.fields}
    out["field_recall"] = len(gnames & enames) / len(gnames)
    out["field_precision"] = len(gnames & enames) / max(1, len(enames))
    if einst is None:
        out["reasons"].append("no instance")
        return out

    def numbers(inst):
        vals = []
        for v in inst.parameter_values.values():
            vals += list(v) if isinstance(v, (list, tuple)) else [v]
        return vals

    gv, ev = numbers(ginst), numbers(einst)
    out["value_recall"] = sum(any(abs(g - e) <= 1e-9 * max(1, abs(g)) for e in ev) for g in gv) / len(gv)

    geom_ok = False
    if einst.geometry is not None:
        a, b = ginst.geometry, einst.geometry
        geom_ok = (len(b.lower_left) == len(a.lower_left)
                   and all(abs(x - y) < 1e-9 for x, y in zip(a.lower_left + a.upper_right, b.lower_left + b.upper_right)))
    out["geom_ok"] = geom_ok
    if not geom_ok:
        out["reasons"].append("domain box differs or is missing")

    best = None
    for mapping in _mappings(gold, ext):
        try:
            r = _score_mapping(gold, ginst, ext, einst, mapping, incompressible, seed, free_scale)
        except Exception as exc:
            r = {"eq_ok": False, "ic_ok": False, "bc_ok": False, "n_matched": -1, "bc_sides_ok": -1, "bc_sides_total": 0,
                 "reasons": [f"scorer error: {type(exc).__name__}: {str(exc)[:100]}"], "eq": [], "eq_modulo": False,
                 "n_extra_eq": 0, "mapping": mapping}
        if best is None or (r["n_matched"], r["bc_sides_ok"]) > (best["n_matched"], best["bc_sides_ok"]):
            best = r
    out.update({k: best[k] for k in ("eq_ok", "eq_modulo", "ic_ok", "bc_ok", "bc_sides_ok", "bc_sides_total",
                                      "n_matched", "n_extra_eq", "mapping")})
    out["n_eq_gold"] = len(gold.equations)
    out["reasons"] += best["reasons"]
    out["semantic_ok"] = bool(out["eq_ok"] and out["ic_ok"] and out["bc_ok"] and geom_ok)
    return out

"""Ground truth for the extraction benchmark: gold PDE-IR + gold ProblemInstance per corpus case.

The gold models come from the repository's hand-encoded `benchmarks/` (the same objects its test-suite
verifies end-to-end); the instances carry the values stated in the corpus documents.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/home/vladimir/development/dealii-agent")

from benchmarks.allen_cahn.allen_cahn import build_model as _allen_cahn  # noqa: E402
from benchmarks.elasticity.elasticity import build_model as _elasticity  # noqa: E402
from benchmarks.heat_conduction_robin.heat_conduction_robin import build_model as _heat  # noqa: E402
from benchmarks.lid_driven_cavity.lid_driven_cavity import build_model as _cavity  # noqa: E402
from benchmarks.poisson.poisson import build_model as _poisson  # noqa: E402
from benchmarks.transient_heat.transient_heat import build_model as _transient  # noqa: E402
from femagent.instance import ProblemInstance, RectangleGeometry  # noqa: E402


def _instance(values, lower, upper, ids, refinements=4, degrees=None, pin=None) -> ProblemInstance:
    return ProblemInstance(
        parameter_values=values,
        geometry=RectangleGeometry(lower_left=lower, upper_right=upper, boundary_ids=ids, refinements=refinements),
        field_degrees=degrees or {},
        pin_fields=pin or [],
    )


GOLD = {
    "heat_nafems": dict(
        model=_heat(),
        instance=_instance({"k": 52.0, "h": 750.0, "T_amb": 0.0}, (0, 0), (0.6, 1.0),
                           {"left": 0, "right": 1, "bottom": 2, "top": 3}),
        time=None,
    ),
    "poisson": dict(
        model=_poisson(),
        instance=_instance({"k": 2.5, "f": 10.0, "g_N": 3.0}, (0, 0), (2.0, 1.0),
                           {"dirichlet_boundary": [0, 2], "neumann_boundary": 1}),
        time=None,
    ),
    "transient_heat": dict(
        model=_transient(),
        instance=_instance({"k": 0.5, "T0": 2.0}, (0, 0), (1.0, 1.0),
                           {"heated_wall": 0, "far_wall": 1, "bottom": 2, "top": 3}),
        time=dict(dt=0.01, num_steps=100),
    ),
    "elasticity": dict(
        model=_elasticity(),
        instance=_instance({"lambda": 1.5, "mu": 1.0, "f": [0.0, 0.0], "g_N": [0.0, -0.05]}, (0, 0), (4.0, 1.0),
                           {"dirichlet_boundary": 0, "traction_boundary": 1}),
        time=None,
    ),
    "stokes": dict(
        model=_cavity(),
        instance=_instance({"nu": 0.1, "u_lid": [1.0, 0.0]}, (0, 0), (1.0, 1.0),
                           {"left": 0, "right": 1, "bottom": 2, "top": 3}, degrees={"u": 2, "p": 1}, pin=["p"]),
        time=None,
        incompressible=True,
        free_scale=(1,),
    ),
    "allen_cahn": dict(
        model=_allen_cahn(),
        instance=_instance({"M": 1.0, "kappa": 0.01}, (0, 0), (1.0, 1.0), {"domain_boundary": 0},
                           refinements=3, degrees={"c": 2}),
        time=dict(dt=0.01, num_steps=100),
    ),
}

CASE_ORDER = ["heat_nafems", "poisson", "transient_heat", "elasticity", "stokes", "allen_cahn"]

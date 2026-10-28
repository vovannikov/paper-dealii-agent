"""Benchmark corpus: paper-style documents with a known ground truth.

Each case is rendered twice from one source: a LaTeX article (-> PDF -> text, the "PDF" modality) and
an HTML page with MathML equations (the "HTML" modality, the format `femagent.extraction.adapters.html`
parses). The ground truth (gold PDE-IR + gold ProblemInstance) lives in `gold.py`.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

import latex2mathml.converter as l2m

# block kinds: ("h", heading) ("p", text with $inline$ math) ("eq", latex)
CASES: dict[str, dict] = {}


def case(case_id: str, title: str, authors: str, blocks: list[tuple]) -> None:
    CASES[case_id] = {"id": case_id, "title": title, "authors": authors, "blocks": blocks}


case(
    "poisson",
    "Steady Diffusion with a Volumetric Source and Mixed Boundary Conditions",
    "A. Reviewer and B. Tester",
    [
        ("h", "1. Problem description"),
        ("p", "We study the steady diffusion of a scalar quantity $u$ (for instance a solute concentration) in a "
              "rectangular domain $\\Omega=(0,2)\\times(0,1)$. The quantity is produced inside the domain at a uniform "
              "volumetric rate $f$, and it is transported by diffusion only; the diffusivity $k$ is constant in space."),
        ("h", "2. Governing equation"),
        ("p", "The balance of the diffusive flux $-k\\nabla u$ against the production rate gives the Poisson equation"),
        ("eq", "-\\nabla\\cdot\\left(k\\,\\nabla u\\right) = f \\quad \\text{in } \\Omega ."),
        ("h", "3. Boundary conditions"),
        ("p", "The quantity vanishes on the left and the bottom side of the domain,"),
        ("eq", "u = 0 \\quad \\text{on } x=0 \\text{ and on } y=0 ."),
        ("p", "Through the right side, $x=2$, a prescribed outward flux $g_N$ leaves the domain,"),
        ("eq", "k\\,\\frac{\\partial u}{\\partial n} = g_N \\quad \\text{on } x=2 ,"),
        ("p", "where $n$ is the outward unit normal. The top side, $y=1$, is impermeable, i.e. "
              "$k\\,\\partial u/\\partial n = 0$ there."),
        ("h", "4. Parameters"),
        ("p", "The diffusivity is $k=2.5$ and the production rate is $f=10$. The outward flux through the right side "
              "is $g_N=3$. All quantities are given in consistent non-dimensional units."),
        ("h", "5. Remarks on the solution"),
        ("p", "For these data the maximum of $u$ is reached close to the top right corner. Earlier studies with a "
              "constant production rate of $f=4$ and a flux of $g_N=1$ reported a maximum of about $0.9$; those "
              "values are not used here."),
    ],
)

case(
    "transient_heat",
    "Transient Heating of a Square Plate with a Ramped Wall Temperature",
    "C. Author and D. Engineer",
    [
        ("h", "1. Problem description"),
        ("p", "We consider the unsteady heat conduction in a thin square plate $\\Omega=(0,1)\\times(0,1)$ over the "
              "time interval $0\\le t\\le 1$. The plate is initially at the reference temperature $T=0$. At $t=0$ the "
              "left edge starts to be heated such that its temperature rises linearly with time; the three other "
              "edges are thermally insulated."),
        ("h", "2. Governing equation"),
        ("p", "With the constant thermal diffusivity $\\kappa$ and no internal heat sources, the temperature $T(x,y,t)$ "
              "obeys the heat equation"),
        ("eq", "\\frac{\\partial T}{\\partial t} - \\nabla\\cdot\\left(\\kappa\\,\\nabla T\\right) = 0 \\quad \\text{in } \\Omega,\\ t>0 ."),
        ("h", "3. Initial and boundary conditions"),
        ("p", "The initial condition is"),
        ("eq", "T(x,y,0) = 0 ."),
        ("p", "On the left edge, $x=0$, the temperature follows the ramp"),
        ("eq", "T(0,y,t) = \\beta\\, t ,"),
        ("p", "where $\\beta$ is the heating rate. The right, bottom and top edges are insulated, so that the normal "
              "heat flux vanishes there:"),
        ("eq", "\\kappa\\,\\frac{\\partial T}{\\partial n} = 0 \\quad \\text{on } x=1,\\ y=0,\\ y=1 ."),
        ("h", "4. Parameters"),
        ("p", "We use $\\kappa=0.5$ and a heating rate of $\\beta=2$. Time is measured in seconds and temperature in "
              "kelvin above the reference temperature. The computations in Section 5 use the time step "
              "$\\Delta t=0.01$."),
        ("h", "5. Notes"),
        ("p", "After the initial transient the interior follows the wall with a constant lag; at $t=1$ the temperature "
              "at the centre of the plate is of the order of one kelvin."),
    ],
)

case(
    "elasticity",
    "Plane-Strain Cantilever Loaded by an End Traction",
    "E. Mechanic and F. Solid",
    [
        ("h", "1. Problem description"),
        ("p", "A short cantilever occupies the rectangle $\\Omega=(0,4)\\times(0,1)$ and is analysed in plane strain. "
              "The beam is clamped at its left end, $x=0$. A uniform shear traction acts downwards on the right end, "
              "$x=4$, while the upper and lower faces are free of traction. Body forces are neglected."),
        ("h", "2. Governing equations"),
        ("p", "The displacement field $u=(u_x,u_y)$ satisfies the equilibrium equation"),
        ("eq", "-\\nabla\\cdot\\sigma(u) = f \\quad \\text{in } \\Omega ,"),
        ("p", "with the body force $f$. The material is isotropic and linearly elastic, with the stress"),
        ("eq", "\\sigma(u) = \\lambda\\,\\mathrm{tr}\\,\\varepsilon(u)\\, I + 2\\mu\\,\\varepsilon(u),"),
        ("p", "and the small-strain tensor"),
        ("eq", "\\varepsilon(u) = \\tfrac{1}{2}\\left(\\nabla u + \\nabla u^{T}\\right) ."),
        ("p", "Here $\\lambda$ and $\\mu$ are the Lam\\'e parameters and $I$ is the identity tensor."),
        ("h", "3. Boundary conditions"),
        ("p", "The clamped end is fixed,"),
        ("eq", "u = 0 \\quad \\text{on } x=0 ,"),
        ("p", "and the prescribed traction on the right end is"),
        ("eq", "\\sigma(u)\\,n = g_N \\quad \\text{on } x=4 ,"),
        ("p", "with $n$ the outward unit normal. On the faces $y=0$ and $y=1$ we have $\\sigma(u)\\,n=0$."),
        ("h", "4. Parameters"),
        ("p", "The Lam\\'e parameters are $\\lambda=1.5$ and $\\mu=1.0$ (non-dimensional). The body force vanishes, "
              "$f=(0,0)$, and the end traction is $g_N=(0,-0.05)$."),
        ("h", "5. Results"),
        ("p", "The tip deflection predicted by Euler--Bernoulli theory for this load is about $-0.09$; the finite "
              "element solution is expected to be slightly larger in magnitude because of shear deformation."),
    ],
)

case(
    "stokes",
    "Creeping Flow in a Lid-Driven Square Cavity",
    "G. Fluid and H. Flow",
    [
        ("h", "1. Problem description"),
        ("p", "We consider the steady flow of an incompressible Newtonian fluid in the unit square cavity "
              "$\\Omega=(0,1)\\times(0,1)$. The upper wall, $y=1$, slides in its own plane with a constant velocity, "
              "whereas the other three walls are at rest. The Reynolds number is so small that inertia is negligible."),
        ("h", "2. Governing equations"),
        ("p", "The velocity $u=(u_x,u_y)$ and the pressure $p$ satisfy the Stokes equations"),
        ("eq", "-\\nabla\\cdot\\left(2\\nu\\,\\varepsilon(u)\\right) + \\nabla p = 0 \\quad \\text{in } \\Omega,"),
        ("eq", "\\nabla\\cdot u = 0 \\quad \\text{in } \\Omega,"),
        ("p", "where $\\nu$ is the kinematic viscosity and $\\varepsilon(u)=\\tfrac12(\\nabla u+\\nabla u^{T})$ is the rate "
              "of strain tensor."),
        ("h", "3. Boundary conditions"),
        ("p", "On the moving lid the velocity is prescribed,"),
        ("eq", "u = u_{\\mathrm{lid}} \\quad \\text{on } y=1 ,"),
        ("p", "and the remaining walls, $x=0$, $x=1$ and $y=0$, are no-slip walls, $u=0$. Because the velocity is "
              "prescribed on the whole boundary, the pressure is determined only up to an additive constant."),
        ("h", "4. Parameters"),
        ("p", "The viscosity is $\\nu=0.1$ and the lid moves in the positive $x$-direction with "
              "$u_{\\mathrm{lid}}=(1,0)$."),
        ("h", "5. Discussion"),
        ("p", "A primary vortex forms slightly above the centre of the cavity; secondary vortices appear in the lower "
              "corners only for much larger cavities."),
    ],
)

case(
    "allen_cahn",
    "Phase Ordering Described by the Allen--Cahn Equation",
    "I. Phase and J. Field",
    [
        ("h", "1. Model"),
        ("p", "The evolution of a non-conserved order parameter $c(x,y,t)$ in the unit square $\\Omega=(0,1)^2$ is "
              "modelled as an $L^2$ gradient flow of the Ginzburg--Landau free energy"),
        ("eq", "F[c] = \\int_\\Omega \\left( f(c) + \\frac{\\kappa}{2}\\,|\\nabla c|^2 \\right) d\\Omega ,"),
        ("p", "with the double-well potential"),
        ("eq", "f(c) = \\frac14\\left(c^2-1\\right)^2 ."),
        ("p", "The resulting Allen--Cahn equation reads"),
        ("eq", "\\frac{\\partial c}{\\partial t} = -M\\left( f'(c) - \\kappa\\,\\nabla^2 c \\right) \\quad \\text{in } \\Omega,\\ t>0 ,"),
        ("p", "where $M$ is the mobility and $\\kappa$ the gradient-energy coefficient."),
        ("h", "2. Boundary and initial conditions"),
        ("p", "No flux crosses the boundary of the domain,"),
        ("eq", "\\nabla c\\cdot n = 0 \\quad \\text{on } \\partial\\Omega ,"),
        ("p", "and the initial state is a smooth saddle-shaped field between the two wells,"),
        ("eq", "c(x,y,0) = \\tfrac12\\,(2x-1)(2y-1) ."),
        ("h", "3. Parameters"),
        ("p", "We use $M=1$ and $\\kappa=0.01$. The simulation runs up to $t=1$ with the time step $\\Delta t=0.01$."),
        ("h", "4. Discussion"),
        ("p", "Because the flow is not conservative the mean of $c$ is not preserved, whereas the free energy $F$ "
              "decreases monotonically in time."),
    ],
)


# -- renderers ------------------------------------------------------------------------------------


def to_latex(c: dict) -> str:
    out = [
        "\\documentclass[11pt,a4paper]{article}",
        "\\usepackage[T1]{fontenc}\\usepackage{amsmath,amssymb}\\usepackage[margin=2.5cm]{geometry}",
        f"\\title{{{c['title']}}}\\author{{{c['authors']}}}\\date{{}}",
        "\\begin{document}\\maketitle",
    ]
    for kind, body, *_ in c["blocks"]:
        if kind == "h":
            out.append(f"\\section*{{{body}}}")
        elif kind == "p":
            out.append(body + "\n")
        elif kind == "eq":
            out.append("\\begin{equation}\n" + body + "\n\\end{equation}")
    out.append("\\end{document}")
    return "\n".join(out)


def _inline_math(text: str) -> str:
    parts = re.split(r"(\$[^$]+\$)", text)
    rendered = []
    for part in parts:
        if part.startswith("$") and part.endswith("$"):
            rendered.append(l2m.convert(_for_mathml(part[1:-1])))
        else:
            rendered.append(html.escape(_plain(part)))
    return "".join(rendered)


def _plain(text: str) -> str:
    return text.replace("--", "\u2013").replace("\\'e", "\u00e9")


def _for_mathml(latex: str) -> str:
    latex = latex.replace("\\tfrac", "\\frac").replace("\\'e", "\u00e9").replace("\\ ", " ")
    latex = latex.replace("\\mathrm{tr}", "\\text{tr}").replace("\\,", " ")
    return latex


def to_html(c: dict) -> str:
    out = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(_plain(c['title']))}</title></head><body><main>",
        f"<h1>{html.escape(_plain(c['title']))}</h1><p class=\"authors\">{html.escape(c['authors'])}</p>",
    ]
    n = 0
    for kind, body, *_ in c["blocks"]:
        if kind == "h":
            out.append(f"<h2>{html.escape(body)}</h2>")
        elif kind == "p":
            out.append(f"<p>{_inline_math(body)}</p>")
        elif kind == "eq":
            n += 1
            mathml = l2m.convert(_for_mathml(body), display="block")
            out.append(f"<div class=\"equation\">{mathml}<span class=\"eqno\">({n})</span></div>")
    out.append("</main></body></html>")
    return "\n".join(out)


def write_all(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for cid, c in CASES.items():
        (directory / f"{cid}.tex").write_text(to_latex(c), encoding="utf-8")
        (directory / f"{cid}.html").write_text(to_html(c), encoding="utf-8")


if __name__ == "__main__":
    import sys

    write_all(Path(sys.argv[1]))
    print("wrote", ", ".join(CASES))

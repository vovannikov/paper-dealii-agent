"""Baseline A of the research plan, at toy scale: LLM -> complete deal.II program (no IR), with compile/run feedback.

For the heat (NAFEMS) page. The model gets the same page text as femagent's extractor and must write main.cc; up to four
attempts, each followed by the first lines of the compiler error or the run failure (the PDEAgent-Bench-style feedback
protocol). Success criterion: the printed probe value at P=(0.6,0.2) within 1e-2 of the published 18.25.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/vladimir/development/dealii-agent")
from openai import OpenAI  # noqa: E402

from femagent.extraction.assemble import render_document  # noqa: E402
from femagent.models.base import Message  # noqa: E402
from femagent.models.openai_compatible import OpenAICompatibleModel  # noqa: E402
from femagent.solve import load_paper  # noqa: E402
from run_extraction import RecordingClient  # noqa: E402

HERE = Path(__file__).parent
DEAL_II = "/home/vladimir/development/tpl/dealii-fork-warpack-sundials"
REF = 18.25
CMAKE = """CMAKE_MINIMUM_REQUIRED(VERSION 3.13.4)
FIND_PACKAGE(deal.II 9.3 QUIET HINTS ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR})
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "deal.II not found")
ENDIF()
DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(baseline)
ADD_EXECUTABLE(baseline main.cc)
DEAL_II_SETUP_TARGET(baseline)
"""
SYSTEM = (
    "You are an expert in the C++ finite element library deal.II (version 9.x, C++17). Write ONE complete, "
    "self-contained program main.cc that solves the problem described in the document below, using continuous Lagrange "
    "elements on a structured quadrilateral mesh. After solving, the program must print exactly one line "
    "`PROBE <value>` with the finite-element solution evaluated at the point named in the document as the reference "
    "point (use VectorTools::point_value). Choose the mesh resolution yourself. Return only the code in a single "
    "```cpp block."
)


def build_and_run(code: str, work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    (work / "main.cc").write_text(code)
    (work / "CMakeLists.txt").write_text(CMAKE)
    build = work / "build"
    subprocess.run(["rm", "-rf", str(build)])
    build.mkdir()
    cm = subprocess.run(["cmake", f"-DDEAL_II_DIR={DEAL_II}", ".."], cwd=build, capture_output=True, text=True)
    if cm.returncode != 0:
        return dict(stage="configure", log=cm.stderr[-1500:])
    mk = subprocess.run(["make", "-j4"], cwd=build, capture_output=True, text=True)
    if mk.returncode != 0:
        errors = [l for l in (mk.stdout + mk.stderr).splitlines() if "error" in l.lower()]
        return dict(stage="compile", log="\n".join(errors[:12]))
    try:
        run = subprocess.run(["./baseline"], cwd=build, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return dict(stage="run", log="timeout after 300 s")
    if run.returncode != 0:
        return dict(stage="run", log=(run.stdout + run.stderr)[-1200:])
    m = re.search(r"PROBE\s+([-+0-9.eE]+)", run.stdout)
    if not m:
        return dict(stage="output", log="no PROBE line in the output:\n" + run.stdout[-600:])
    return dict(stage="ok", value=float(m.group(1)))


def extract_code(text: str) -> str:
    m = re.search(r"```(?:cpp|c\+\+)?\s*(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip()


def run_model(model_name: str, case: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    doc = load_paper(HERE / "corpus" / f"{case}.html")
    client = RecordingClient(OpenAI(base_url="http://localhost:11434/v1", api_key="x", timeout=2400, max_retries=0))
    llm = OpenAICompatibleModel(model_name, base_url="http://localhost:11434/v1", client=client)
    messages = [Message("system", SYSTEM), Message("user", render_document(doc))]
    attempts = []
    t0 = time.time()
    for k in range(1, 5):
        client.stage = f"attempt{k}"
        try:
            raw = llm.complete(messages, temperature=0.0, max_tokens=12000)
        except Exception as exc:
            attempts.append(dict(k=k, stage="llm_error", log=str(exc)[:300]))
            break
        code = extract_code(raw)
        result = build_and_run(code, out / f"attempt{k}")
        result["k"] = k
        if result["stage"] == "ok":
            result["rel_error"] = abs(result["value"] - REF) / REF
        attempts.append(result)
        if result["stage"] == "ok":
            break
        messages += [Message("assistant", raw), Message(
            "user", f"The program failed at the {result['stage']} stage:\n{result['log']}\nReturn the corrected complete program.")]
    calls = client.calls
    summary = dict(model=model_name, case=case, attempts=attempts, seconds=time.time() - t0,
                   prompt_tokens=sum(c.get("prompt_tokens") or 0 for c in calls),
                   completion_tokens=sum(c.get("completion_tokens") or 0 for c in calls), llm_calls=len(calls))
    (out / "baseline.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    for name in sys.argv[1:]:
        tag = name.replace(":", "_")
        s = run_model(name, "heat_nafems", HERE / "baseline_runs" / tag)
        last = s["attempts"][-1]
        print(name, [(a["stage"], round(a.get("rel_error", -1), 5)) for a in s["attempts"]], f"{s['prompt_tokens']}+{s['completion_tokens']} tok", f"{s['seconds']:.0f}s", flush=True)

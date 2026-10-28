"""Stage 1: ask each LLM to extract the PDE model and the problem instance from each corpus document.

Mirrors `femagent solve` up to (and excluding) code generation: same prompts, same validation-repair loop,
temperature 0. Everything is recorded for the analysis: raw answers, reasoning, per-call latency and tokens.
Resumable: a run directory with `record.json` is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, "/home/vladimir/development/dealii-agent")

from openai import OpenAI  # noqa: E402

from femagent.extraction.assemble import extract_pde_model, extract_problem_instance, render_document  # noqa: E402
from femagent.models.openai_compatible import OpenAICompatibleModel  # noqa: E402
from femagent.solve import _save_attempts, load_paper  # noqa: E402

HERE = Path(__file__).parent
BASE_URL = "http://localhost:11434/v1"

ADDENDUM_V2 = (
    "\n\nAdditional node types and rules (all nodes are objects tagged by \"type\"):\n"
    '- {"type": "TimeCoordinate"} is the time variable t. Use it for time-dependent data (e.g. a ramp beta*t); '
    "never declare t as a parameter.\n"
    '- {"type": "SpatialCoordinate", "axis": 0} is x, axis 1 is y.\n'
    '- {"type": "VectorExpr", "components": [<Expr>, ...]} builds a vector from scalar components; Field and Parameter '
    'have "rank": "scalar", "vector" or "tensor". A vector Dirichlet value of zero is a VectorExpr of Constant 0.\n'
    '- Unary nodes Transpose, Trace take "operand"; {"type": "Identity"} is the identity tensor; '
    '{"type": "Convection", "velocity": <Expr>, "operand": <Expr>} is (velocity . grad) operand.\n'
    '- Every Equation needs "name", "lhs" and "rhs"; Apply needs "name", "args" and "rank"; Pow needs "base" and '
    '"exponent" (both Expr, e.g. a Constant).\n'
    '- Initial data: "initial_conditions": [{"field": "c", "value": <Expr of coordinates>}]; give it whenever the '
    "paper states an initial state.\n"
    '- Named constitutive laws: "constitutive_relations": [{"name": "f_prime", "definition": {"type": "Equation", '
    '"name": "f_prime", "lhs": {"type": "Apply", "name": "f_prime", "args": [<Field c>], "rank": "scalar"}, '
    '"rhs": <body Expr>}}]; the equations then use the matching Apply node. Writing the expression inline is also allowed.\n'
    '- Optional metadata ("provenance", "ambiguity") must be null unless you can fill it exactly; omit it otherwise.\n'
)

ADDENDUM_V3 = ADDENDUM_V2.replace(
    " A vector Dirichlet value of zero is a VectorExpr of Constant 0.",
    ' A zero vector value (clamped end, no-slip wall) is simply {"type": "Constant", "value": 0}.',
) + '- Declare every field with "space": "H1" (the finite-element spaces are chosen later by the solver).\n'


class RecordingClient:
    """Wraps an OpenAI client and logs latency and token usage of every chat completion."""

    def __init__(self, client: OpenAI) -> None:
        self._client = client
        self.calls: list[dict] = []
        self.stage = ""
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        start = time.time()
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # keep the failure in the record, then let the pipeline see it
            self.calls.append(dict(stage=self.stage, seconds=time.time() - start, error=repr(exc)[:300]))
            raise
        usage = response.usage
        choice = response.choices[0]
        reasoning = getattr(choice.message, "reasoning", None) or getattr(choice.message, "reasoning_content", None) or ""
        self.calls.append(dict(
            stage=self.stage, seconds=time.time() - start,
            prompt_tokens=getattr(usage, "prompt_tokens", None), completion_tokens=getattr(usage, "completion_tokens", None),
            answer_chars=len(choice.message.content or ""), reasoning_chars=len(reasoning), finish=choice.finish_reason,
        ))
        return response


def _issues(generation) -> list[str]:
    return [f"{i.code}: {i.message}" for i in generation.issues]


def _first_attempt_ok(generation) -> bool:
    first = generation.attempts[0] if generation.attempts else None
    return bool(first and first.parse_error is None and not any(i.severity.value == "error" for i in first.issues))


def run_one(model_name: str, case: str, modality: str, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    source = HERE / "corpus" / (f"{case}.html" if modality == "html" else f"{case}.txt")
    document = load_paper(source)
    prompt_chars = len(render_document(document)) if not isinstance(document, str) else len(document)

    client = RecordingClient(OpenAI(base_url=BASE_URL, api_key="not-needed", timeout=2400, max_retries=0))
    llm = OpenAICompatibleModel(model_name, base_url=BASE_URL, client=client)
    record = dict(model=model_name, case=case, modality=modality, document_chars=prompt_chars, started=time.time())

    client.stage = "model"
    t0 = time.time()
    try:
        extracted = extract_pde_model(llm, document)
    except Exception as exc:  # transport failure (e.g. timeout)
        record.update(model_ok=False, model_error=repr(exc)[:300], total_seconds=time.time() - t0)
        record["calls"] = client.calls
        (out / "record.json").write_text(json.dumps(record, indent=1))
        return record
    record.update(
        model_ok=extracted.succeeded, model_attempts=len(extracted.attempts), model_first_ok=_first_attempt_ok(extracted),
        model_issues=_issues(extracted), model_seconds=time.time() - t0,
        model_parse_errors=[a.parse_error for a in extracted.attempts if a.parse_error],
    )
    _save_attempts(out, "model", extracted)
    if extracted.value is not None:
        (out / "model.json").write_text(extracted.value.model_dump_json(indent=2))

    if extracted.succeeded:
        client.stage = "instance"
        t1 = time.time()
        try:
            inst = extract_problem_instance(extracted.value, document, llm)
        except Exception as exc:
            record.update(instance_ok=False, instance_error=repr(exc)[:300])
        else:
            record.update(
                instance_ok=inst.succeeded, instance_attempts=len(inst.attempts), instance_first_ok=_first_attempt_ok(inst),
                instance_issues=_issues(inst), instance_seconds=time.time() - t1,
            )
            _save_attempts(out, "instance", inst)
            if inst.value is not None:
                (out / "instance.json").write_text(inst.value.model_dump_json(indent=2))
    record["total_seconds"] = time.time() - t0
    record["calls"] = client.calls
    (out / "record.json").write_text(json.dumps(record, indent=1))
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--cases", nargs="+", default=["heat_nafems", "poisson", "transient_heat", "elasticity", "stokes", "allen_cahn"])
    parser.add_argument("--modalities", nargs="+", default=["html", "pdf"])
    parser.add_argument("--runs", default=str(HERE / "runs"))
    parser.add_argument("--prompt", default="v1", choices=["v1", "v2", "v3"])
    args = parser.parse_args()
    if args.prompt in ("v2", "v3"):
        import femagent.extraction.assemble as assemble

        original = assemble._system_prompt
        assemble._system_prompt = lambda: original() + (ADDENDUM_V3 if args.prompt == "v3" else ADDENDUM_V2)
    for model in args.models:
        tag = model.replace(":", "_").replace("/", "_")
        for modality in args.modalities:
            for case in args.cases:
                out = Path(args.runs) / tag / modality / case
                if (out / "record.json").exists():
                    continue
                ext = "html" if modality == "html" else "txt"
                if not (HERE / "corpus" / f"{case}.{ext}").exists():
                    continue
                print(f"[{time.strftime('%H:%M:%S')}] {model} {modality} {case} ...", flush=True)
                rec = run_one(model, case, "html" if modality == "html" else "pdf", out)
                print(f"    model_ok={rec.get('model_ok')} instance_ok={rec.get('instance_ok')} "
                      f"attempts={rec.get('model_attempts')}/{rec.get('instance_attempts')} total={rec.get('total_seconds', 0):.0f}s", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Readable GEPA example for the contract auto-renewal task.

This script keeps the same seed prompt and the same dataset as `gepa-1.py`,
but hands the search loop over to the original GEPA library.

Why this file exists:
- `gepa-1.py` shows a hand-rolled GEPA-like loop.
- this file shows the minimum pieces you need to plug a custom task into
  the real GEPA engine
- the adapter is intentionally small and explicit so it is easier to learn from

Run with:
    python gepa_original_library.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
GEPA_SRC = BASE_DIR / "gepa" / "src"

if str(GEPA_SRC) not in sys.path:
    sys.path.insert(0, str(GEPA_SRC))

from gepa.api import optimize
from gepa.core.adapter import EvaluationBatch, GEPAAdapter


load_dotenv(BASE_DIR / ".env")

TASK_MODEL = "claude-haiku-4-5"
JUDGE_MODEL = "claude-haiku-4-5"
MUTATION_MODEL = "claude-haiku-4-5"

RUN_DIR = BASE_DIR / "outputs" / "gepa_original_library"

SEED_CANDIDATE = {
    "system_prompt": """
You are an information extraction system for contracts.
Return a binary answer and a brief justification grounded in the text.
""".strip(),
    "user_template": "TEXT:\n{{TEXT}}",
}

DATASET = [
    {
        "text": "This Agreement shall automatically renew for successive one-year terms unless either party gives written notice of non-renewal at least 30 days before the end of the current term.",
        "true_answer": "Yes",
        "true_explanation": "The agreement renews automatically unless notice of non-renewal is given.",
    },
    {
        "text": "The term of this lease shall expire on June 30, 2026. Any extension or renewal must be agreed to in writing by both parties.",
        "true_answer": "No",
        "true_explanation": "Renewal requires written agreement, so there is no automatic renewal.",
    },
    {
        "text": "This subscription remains in force for 12 months. At the end of the term, the customer may renew by submitting a renewal order.",
        "true_answer": "No",
        "true_explanation": "The customer may renew, but renewal requires an affirmative step, so it is not automatic.",
    },
    {
        "text": "This Agreement will renew automatically each month unless terminated by either party with 15 days' notice.",
        "true_answer": "Yes",
        "true_explanation": "The agreement renews automatically each month unless terminated with notice.",
    },
    {
        "text": "The initial term is one year, and this agreement will automatically extend for additional one-year periods unless either party gives notice of non-renewal.",
        "true_answer": "Yes",
        "true_explanation": "The text says the agreement automatically extends unless notice is given.",
    },
    {
        "text": "This contract ends at the close of the initial term. Any continuation must be set out in a later signed amendment.",
        "true_answer": "No",
        "true_explanation": "Continuation requires a later signed amendment, so renewal is not automatic.",
    },
    {
        "text": "Each monthly service term renews automatically on the billing date unless the customer cancels before that date.",
        "true_answer": "Yes",
        "true_explanation": "The service renews automatically unless the customer cancels.",
    },
    {
        "text": "Upon expiration, the customer may choose to purchase another annual term by sending a new purchase order.",
        "true_answer": "No",
        "true_explanation": "Renewal depends on a new purchase order, so it is not automatic.",
    },
]

TASK_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "extraction_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "enum": ["Yes", "No"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "reasoning"],
            "additionalProperties": False,
        },
    },
}

JUDGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning_score": {"type": "integer", "enum": [0, 1, 2]},
                "diagnosis": {"type": "string"},
                "suggested_fix": {"type": "string"},
            },
            "required": ["reasoning_score", "diagnosis", "suggested_fix"],
            "additionalProperties": False,
        },
    },
}

MUTATION_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "mutation_result",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "diagnosis_summary": {"type": "string"},
                "system_prompt": {"type": "string"},
                "user_template": {"type": "string"},
            },
            "required": ["diagnosis_summary", "system_prompt", "user_template"],
            "additionalProperties": False,
        },
    },
}


def response_format_instructions(response_format: dict[str, Any]) -> str:
    schema = response_format["json_schema"]["schema"]
    return (
        "Return JSON only. Do not wrap it in markdown.\n"
        "Follow this JSON schema exactly:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def extract_text_from_anthropic_response(response: Any) -> str:
    parts: list[str] = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    return "\n".join(parts).strip()


def extract_json_from_text(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model did not return JSON: {text}")

    return text[start : end + 1]


def render_user_message(user_template: str, text: str) -> str:
    return user_template.replace("{{TEXT}}", text)


class ContractRenewalGEPAAdapter(GEPAAdapter[dict[str, str], dict[str, Any], dict[str, Any]]):
    """
    A small custom adapter that shows the three moving pieces GEPA needs:

    1. `evaluate`:
       run the current candidate on a batch and assign scores
    2. `make_reflective_dataset`:
       convert execution traces into compact, high-signal reflection examples
    3. `propose_new_texts`:
       ask a reflection model to rewrite the prompt components
    """

    def __init__(self, task_model: str, judge_model: str, mutation_model: str):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is missing. Add it to prompt_optim_gepa/.env before running."
            )

        self.client = Anthropic(api_key=api_key)
        self.task_model = task_model
        self.judge_model = judge_model
        self.mutation_model = mutation_model

    def _chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
    ) -> dict[str, Any]:
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, str]] = []

        for message in messages:
            if message["role"] == "system":
                system_parts.append(message["content"])
            else:
                anthropic_messages.append(message)

        system_parts.append(response_format_instructions(response_format))

        completion = self.client.messages.create(
            model=model,
            system="\n\n".join(system_parts),
            messages=anthropic_messages,
            max_tokens=1200,
        )

        text = extract_text_from_anthropic_response(completion)
        return json.loads(extract_json_from_text(text))

    def predict(self, candidate: dict[str, str], text: str) -> dict[str, str]:
        messages = [
            {"role": "system", "content": candidate["system_prompt"]},
            {
                "role": "user",
                "content": render_user_message(candidate["user_template"], text),
            },
        ]

        out = self._chat_json(
            model=self.task_model,
            messages=messages,
            response_format=TASK_RESPONSE_FORMAT,
        )

        answer = out.get("answer", "No")
        if answer not in {"Yes", "No"}:
            answer = "No"

        return {
            "answer": answer,
            "reasoning": out.get("reasoning", ""),
        }

    def judge(self, example: dict[str, str], prediction: dict[str, str]) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": """
You are evaluating an extraction result whose task is:
- Decide whether the text contains an automatic renewal clause.
- Answer "Yes" when the agreement renews automatically unless someone opts out or terminates.
- Answer "No" when renewal requires a new order, signature, amendment, or any other affirmative step.

Scoring rubric:
- 2 = reasoning is correct, specific, and grounded
- 1 = reasoning is partly acceptable but vague or incomplete
- 0 = reasoning is wrong, unsupported, contradictory, or too weak

Rules:
- If the predicted answer is wrong, reasoning_score should usually be 0.
- Do not require exact wording match.
- suggested_fix must describe a prompt-level improvement, not a document-specific fact.
""".strip(),
            },
            {
                "role": "user",
                "content": f"""
TEXT:
{example["text"]}

GROUND TRUTH:
answer: {example["true_answer"]}
explanation: {example["true_explanation"]}

MODEL OUTPUT:
answer: {prediction["answer"]}
reasoning: {prediction["reasoning"]}
""".strip(),
            },
        ]

        out = self._chat_json(
            model=self.judge_model,
            messages=messages,
            response_format=JUDGE_RESPONSE_FORMAT,
        )

        if prediction["answer"] != example["true_answer"]:
            out["reasoning_score"] = 0

        return out

    def evaluate_example(self, candidate: dict[str, str], example: dict[str, str]) -> dict[str, Any]:
        try:
            prediction = self.predict(candidate, example["text"])
            judge_out = self.judge(example, prediction)
        except Exception as exc:
            prediction = {
                "answer": "No",
                "reasoning": "",
            }
            judge_out = {
                "reasoning_score": 0,
                "diagnosis": f"Evaluation failed: {type(exc).__name__}: {exc}",
                "suggested_fix": "Make the prompt more explicit and stable so the model returns valid structured output.",
            }

        answer_score = 1.0 if prediction["answer"] == example["true_answer"] else 0.0
        reasoning_score = float(judge_out["reasoning_score"]) / 2.0
        total_score = answer_score + reasoning_score

        return {
            "text": example["text"],
            "true_answer": example["true_answer"],
            "true_explanation": example["true_explanation"],
            "pred_answer": prediction["answer"],
            "pred_reasoning": prediction["reasoning"],
            "answer_score": answer_score,
            "reasoning_score": reasoning_score,
            "total_score": total_score,
            "diagnosis": judge_out["diagnosis"],
            "suggested_fix": judge_out["suggested_fix"],
        }

    def evaluate(
        self,
        batch: list[dict[str, str]],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, Any], dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        scores: list[float] = []
        trajectories: list[dict[str, Any]] | None = [] if capture_traces else None
        objective_scores: list[dict[str, float]] = []

        for example in batch:
            result = self.evaluate_example(candidate, example)

            outputs.append(
                {
                    "answer": result["pred_answer"],
                    "reasoning": result["pred_reasoning"],
                }
            )
            scores.append(result["total_score"])
            objective_scores.append(
                {
                    "answer": result["answer_score"],
                    "reasoning": result["reasoning_score"],
                }
            )

            if trajectories is not None:
                trajectories.append(result)

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[dict[str, Any], dict[str, Any]],
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        trajectories = eval_batch.trajectories or []

        examples: list[dict[str, Any]] = []
        for traj in trajectories:
            examples.append(
                {
                    "Inputs": {
                        "text": traj["text"],
                        "gold_answer": traj["true_answer"],
                        "gold_explanation": traj["true_explanation"],
                    },
                    "Generated Outputs": {
                        "answer": traj["pred_answer"],
                        "reasoning": traj["pred_reasoning"],
                    },
                    "Feedback": "\n".join(
                        [
                            f"answer_score: {traj['answer_score']}",
                            f"reasoning_score: {traj['reasoning_score']}",
                            f"diagnosis: {traj['diagnosis']}",
                            f"suggested_fix: {traj['suggested_fix']}",
                        ]
                    ),
                }
            )

        return {name: examples for name in components_to_update}

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, list[dict[str, Any]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        # `module_selector="all"` means both prompt fields are usually updated together.
        records = reflective_dataset.get("system_prompt")
        if not records:
            records = next(iter(reflective_dataset.values()))

        messages = [
            {
                "role": "system",
                "content": """
You are improving a prompt candidate for a structured extraction task.

Goals:
- fix the failures shown in the reflective dataset
- preserve what already works
- keep the prompt simple and explicit

Rules:
- The task is binary classification of automatic renewal clauses.
- Answer "Yes" only when renewal happens automatically unless someone opts out or terminates.
- Answer "No" when renewal requires any affirmative step such as a new order, signature, amendment, or written agreement.
- Do not change the response format expected by the caller.
- The user_template must contain {{TEXT}} exactly once.
""".strip(),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_candidate": candidate,
                        "components_to_update": components_to_update,
                        "reflective_examples": records,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]

        out = self._chat_json(
            model=self.mutation_model,
            messages=messages,
            response_format=MUTATION_RESPONSE_FORMAT,
        )

        new_system_prompt = out["system_prompt"].strip() or candidate["system_prompt"]
        new_user_template = out["user_template"].strip()

        if new_user_template.count("{{TEXT}}") != 1:
            new_user_template = candidate["user_template"]

        proposed: dict[str, str] = {}

        if "system_prompt" in components_to_update:
            proposed["system_prompt"] = new_system_prompt

        if "user_template" in components_to_update:
            proposed["user_template"] = new_user_template

        return proposed


def summarize_candidate(
    adapter: ContractRenewalGEPAAdapter,
    candidate: dict[str, str],
    dataset: list[dict[str, str]],
) -> dict[str, Any]:
    results = [adapter.evaluate_example(candidate, example) for example in dataset]

    answer_perf = sum(item["answer_score"] for item in results) / len(results)
    reasoning_perf = sum(item["reasoning_score"] for item in results) / len(results)
    total_perf = sum(item["total_score"] for item in results) / len(results)

    return {
        "results": results,
        "answer_perf": answer_perf,
        "reasoning_perf": reasoning_perf,
        "total_perf": total_perf,
    }


def print_summary(label: str, summary: dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(label)
    print(f"answer_perf   : {summary['answer_perf']:.3f}")
    print(f"reasoning_perf: {summary['reasoning_perf']:.3f}")
    print(f"total_perf    : {summary['total_perf']:.3f}")


def main() -> None:
    adapter = ContractRenewalGEPAAdapter(
        task_model=TASK_MODEL,
        judge_model=JUDGE_MODEL,
        mutation_model=MUTATION_MODEL,
    )

    baseline = summarize_candidate(adapter, SEED_CANDIDATE, DATASET)
    print_summary("SEED CANDIDATE", baseline)

    result = optimize(
        seed_candidate=SEED_CANDIDATE,
        trainset=DATASET,
        valset=DATASET,
        adapter=adapter,
        candidate_selection_strategy="pareto",
        module_selector="all",
        reflection_minibatch_size=4,
        frontier_type="instance",
        perfect_score=2.0,
        max_metric_calls=32,
        run_dir=str(RUN_DIR),
        cache_evaluation=True,
        track_best_outputs=False,
        seed=42,
    )

    best_candidate = result.best_candidate
    assert isinstance(best_candidate, dict)

    optimized = summarize_candidate(adapter, best_candidate, DATASET)
    print_summary("BEST CANDIDATE", optimized)

    print("\n" + "#" * 80)
    print("BEST CANDIDATE INDEX:", result.best_idx)
    print("TOTAL CANDIDATES    :", result.num_candidates)
    print("\nBEST SYSTEM PROMPT:\n")
    print(best_candidate["system_prompt"])
    print("\nBEST USER TEMPLATE:\n")
    print(best_candidate["user_template"])
    print("\nRUN DIRECTORY:\n")
    print(RUN_DIR)


if __name__ == "__main__":
    main()

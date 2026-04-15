"""
GEPA prototype v2.

Key changes relative to `gepa-1.py`:
- uses the OpenAI SDK via chat completions
- samples `n` task outputs per example to expose label ambiguity
- scores per-example conviction with Leik's D over an ordered label space
- reports dataset-level precision / recall / F1 for the binary task

The code is intentionally kept close to `gepa-1.py` so the differences are easy
to compare side by side.
"""

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ConfigDict


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

ORDERED_LABELS = ["No", "Yes"]
POSITIVE_LABEL = "Yes"

TASK_MODEL = os.environ.get("OPENAI_TASK_MODEL", "gpt-5.4-nano")
JUDGE_MODEL = os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.4-nano")
MUTATE_MODEL = os.environ.get("OPENAI_MUTATE_MODEL", "gpt-5.4-nano")

TASK_SAMPLES = 5
TASK_TEMPERATURE = 1
JUDGE_TEMPERATURE = 0.2
MUTATE_TEMPERATURE = 0.4

LOG_PATH = BASE_DIR / "gepa2_openai_runs.jsonl"
MINIBATCH_SIZE = 4

_client = None


class TaskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Literal["No", "Yes"]
    reasoning: str


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning_score: Literal[0, 1, 2]
    diagnosis: str
    suggested_fix: str
    include_in_reflection: bool


class MutationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    diagnosis_summary: str
    system_prompt: str
    user_template: str


def get_client():
    global _client

    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is missing. Add it to prompt_optim_gepa/.env before running."
            )
        _client = OpenAI(api_key=api_key)

    return _client


def make_initial_candidate():
    return {
        "system_prompt": """
You are an information extraction system for contracts.
Return a binary answer and a brief justification grounded in the text.
""".strip(),
        "user_template": "TEXT:\n{{TEXT}}",
    }


def render_user_message(user_template, text):
    return user_template.replace("{{TEXT}}", text)


def make_openai_response_format(schema_model, schema_name):
    schema = schema_model.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "strict": True,
            "schema": schema,
        },
    }


def chat_parse_choices(model, messages, response_format, schema_name, temperature, n=1, independent_samples=False):
    def do_request(num_choices):
        return get_client().chat.completions.create(
            model=model,
            messages=messages,
            response_format=make_openai_response_format(response_format, schema_name),
            temperature=temperature,
            n=num_choices,
            max_completion_tokens=1200,
        )

    if independent_samples and n > 1:
        parsed = []
        for _ in range(n):
            completion = do_request(1)
            parsed.append(response_format.model_validate_json(completion.choices[0].message.content or ""))
        return parsed

    try:
        completion = do_request(n)
        parsed = [
            response_format.model_validate_json(choice.message.content or "")
            for choice in completion.choices
        ]
    except Exception:
        if n == 1:
            raise
        parsed = []
        for _ in range(n):
            completion = do_request(1)
            parsed.append(response_format.model_validate_json(completion.choices[0].message.content or ""))

    return parsed


def candidate_hash(candidate):
    canonical = json.dumps(candidate, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def example_hash(example):
    canonical = json.dumps(example, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def label_counts(samples):
    counts = {label: 0 for label in ORDERED_LABELS}
    for sample in samples:
        counts[sample["answer"]] += 1
    return counts


def label_distribution(counts):
    total = sum(counts.values())
    if total == 0:
        return {label: 0.0 for label in ORDERED_LABELS}
    return {label: counts[label] / total for label in ORDERED_LABELS}


def majority_label(counts):
    return max(ORDERED_LABELS, key=lambda label: (counts[label], -ORDERED_LABELS.index(label)))


def leik_d(counts):
    # Leik's D is an ordinal dispersion measure. Lower dispersion means the model
    # is concentrating its samples in a narrower part of the ordered label space.
    if len(ORDERED_LABELS) <= 1:
        return 0.0

    total = sum(counts.values())
    if total == 0:
        return 0.0

    cumulative = 0.0
    numerator = 0.0

    for label in ORDERED_LABELS[:-1]:
        cumulative += counts[label] / total
        numerator += min(cumulative, 1.0 - cumulative)

    return (2.0 * numerator) / (len(ORDERED_LABELS) - 1)


def representative_reasonings(samples, label, limit=2):
    reasonings = []
    for sample in samples:
        if sample["answer"] == label and sample["reasoning"]:
            reasonings.append(sample["reasoning"])
            if len(reasonings) == limit:
                break
    return reasonings


def text_preview(text, limit=72):
    single_line = " ".join(text.split())
    if len(single_line) <= limit:
        return single_line
    return single_line[: limit - 3] + "..."


def format_counts(counts):
    return "/".join(f"{label}={counts[label]}" for label in ORDERED_LABELS)


def log_eval_result(result, prefix="EVAL"):
    sample_total = sum(result["label_counts"].values())
    if sample_total == 0:
        print(
            f"[{prefix}] ERROR "
            f"pred={result['pred_answer']} "
            f"gold={result['true_answer']} "
            f"reason={result['reasoning_score']:.2f} "
            f"diagnosis=\"{text_preview(result['diagnosis'], limit=120)}\" "
            f"text=\"{text_preview(result['text'])}\""
        )
        return

    print(
        f"[{prefix}] "
        f"pred={result['pred_answer']} "
        f"gold={result['true_answer']} "
        f"n={sample_total} "
        f"labels={result['sample_labels']} "
        f"counts={format_counts(result['label_counts'])} "
        f"support={result['gold_support']:.2f} "
        f"conviction={result['conviction_score']:.2f} "
        f"reason={result['reasoning_score']:.2f} "
        f"text=\"{text_preview(result['text'])}\""
    )


def print_metric_block(title, summary):
    print(title)
    print(f"  F1         : {summary['f1']:.3f}")
    print(f"  Recall     : {summary['recall']:.3f}")
    print(f"  Precision  : {summary['precision']:.3f}")
    print(f"  Support    : {summary['gold_support_perf']:.3f}")
    print(f"  Conviction : {summary['conviction_perf']:.3f}")
    print(f"  Reasoning  : {summary['reasoning_perf']:.3f}")


def print_step_banner(step):
    print("\n" + "=" * 80)
    print(f"STEP {step}")


def predict(candidate, text, sample_count=TASK_SAMPLES):
    messages = [
        {"role": "system", "content": candidate["system_prompt"]},
        {"role": "user", "content": render_user_message(candidate["user_template"], text)},
    ]

    raw_outputs = chat_parse_choices(
        model=TASK_MODEL,
        messages=messages,
        response_format=TaskOutput,
        schema_name="extraction_result",
        temperature=TASK_TEMPERATURE,
        n=sample_count,
        independent_samples=True,
    )

    samples = []
    for output in raw_outputs:
        samples.append(
            {
                "answer": output.answer,
                "reasoning": output.reasoning,
            }
        )

    counts = label_counts(samples)
    prediction = majority_label(counts)
    majority_reasonings = representative_reasonings(samples, prediction, limit=1)
    conviction_score = 1.0 - leik_d(counts)

    return {
        "samples": samples,
        "sample_labels": [sample["answer"] for sample in samples],
        "label_counts": counts,
        "label_distribution": label_distribution(counts),
        "majority_answer": prediction,
        "majority_reasoning": majority_reasonings[0] if majority_reasonings else "",
        "conviction_score": conviction_score,
    }


def judge(example, pred):
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
- If the majority predicted answer is wrong, reasoning_score should usually be 0.
- If the same text produced both labels, mention the ambiguity in diagnosis.
- conviction_score ranges from 0.0 to 1.0.
- A conviction_score near 1.0 means the sampled outputs concentrated on one label.
- A conviction_score near 0.0 means the sampled outputs were highly split across the ordered label space.
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

SAMPLED LABEL COUNTS:
{json.dumps(pred["label_counts"], indent=2)}

LABEL DISTRIBUTION:
{json.dumps(pred["label_distribution"], indent=2)}

CONVICTION SCORE:
{round(pred["conviction_score"], 3)}

MAJORITY OUTPUT:
answer: {pred["majority_answer"]}
reasoning: {pred["majority_reasoning"]}

REPRESENTATIVE "YES" REASONINGS:
{json.dumps(representative_reasonings(pred["samples"], "Yes"), ensure_ascii=False, indent=2)}

REPRESENTATIVE "NO" REASONINGS:
{json.dumps(representative_reasonings(pred["samples"], "No"), ensure_ascii=False, indent=2)}
""".strip(),
        },
    ]

    out = chat_parse_choices(
        model=JUDGE_MODEL,
        messages=messages,
        response_format=JudgeOutput,
        schema_name="judge_result",
        temperature=JUDGE_TEMPERATURE,
        n=1,
    )[0].model_dump()

    if pred["majority_answer"] != example["true_answer"]:
        out["reasoning_score"] = 0

    return out


def evaluate_example(candidate, example, log_prefix="EVAL"):
    try:
        pred = predict(candidate, example["text"])
        judge_out = judge(example, pred)
    except Exception as exc:
        fallback_counts = {label: 0 for label in ORDERED_LABELS}
        pred = {
            "samples": [],
            "sample_labels": [],
            "label_counts": fallback_counts,
            "label_distribution": label_distribution(fallback_counts),
            "majority_answer": ORDERED_LABELS[0],
            "majority_reasoning": "",
            "conviction_score": 0.0,
        }
        judge_out = {
            "reasoning_score": 0,
            "diagnosis": f"Evaluation failed: {type(exc).__name__}: {exc}",
            "suggested_fix": "Make the prompt and output instructions more explicit so the model responds consistently with valid structured output.",
            "include_in_reflection": True,
        }

    sample_total = max(1, sum(pred["label_counts"].values()))
    gold_support = pred["label_counts"][example["true_answer"]] / sample_total
    majority_correct = 1.0 if pred["majority_answer"] == example["true_answer"] else 0.0

    result = {
        "example_hash": example_hash(example),
        "text": example["text"],
        "true_answer": example["true_answer"],
        "true_explanation": example["true_explanation"],
        "samples": pred["samples"],
        "sample_labels": pred["sample_labels"],
        "label_counts": pred["label_counts"],
        "label_distribution": pred["label_distribution"],
        "pred_answer": pred["majority_answer"],
        "pred_reasoning": pred["majority_reasoning"],
        "majority_correct": majority_correct,
        "gold_support": gold_support,
        "conviction_score": pred["conviction_score"],
        "reasoning_score": float(judge_out["reasoning_score"]) / 2.0,
        "diagnosis": judge_out["diagnosis"],
        "suggested_fix": judge_out["suggested_fix"],
        "include_in_reflection": bool(judge_out["include_in_reflection"]),
    }

    if log_prefix:
        log_eval_result(result, prefix=log_prefix)

    return result


def precision_recall_f1(results):
    # These dataset metrics are currently binary-task metrics keyed to
    # `POSITIVE_LABEL`. The label-distribution machinery above is already ordinal,
    # which lets us keep conviction generic even before the main task expands.
    tp = fp = tn = fn = 0

    for result in results:
        pred_positive = result["pred_answer"] == POSITIVE_LABEL
        true_positive = result["true_answer"] == POSITIVE_LABEL

        if pred_positive and true_positive:
            tp += 1
        elif pred_positive and not true_positive:
            fp += 1
        elif (not pred_positive) and true_positive:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def summarize_results(results):
    metrics = precision_recall_f1(results)
    majority_total = sum(result["majority_correct"] for result in results)
    support_total = sum(result["gold_support"] for result in results)
    conviction_total = sum(result["conviction_score"] for result in results)
    reasoning_total = sum(result["reasoning_score"] for result in results)

    return {
        "results": results,
        "majority_total": majority_total,
        "support_total": support_total,
        "conviction_total": conviction_total,
        "reasoning_total": reasoning_total,
        "majority_accuracy": majority_total / len(results),
        "gold_support_perf": support_total / len(results),
        "conviction_perf": conviction_total / len(results),
        "reasoning_perf": reasoning_total / len(results),
        **metrics,
    }


def summary_sort_key(summary):
    return (
        summary["f1"],
        summary["recall"],
        summary["precision"],
        summary["gold_support_perf"],
        summary["reasoning_perf"],
        summary["conviction_perf"],
        summary["majority_accuracy"],
    )


def evaluate_full_dataset(candidate, dataset, cached_results_by_example=None, log_prefix="FULL"):
    cached_results_by_example = cached_results_by_example or {}
    results = []

    for example in dataset:
        ex_hash = example_hash(example)
        if ex_hash in cached_results_by_example:
            result = dict(cached_results_by_example[ex_hash])
        else:
            result = evaluate_example(candidate, example, log_prefix=log_prefix)

        if ex_hash in cached_results_by_example and log_prefix:
            log_eval_result(result, prefix=f"{log_prefix}:cached")

        results.append(result)

    return summarize_results(results)


def example_severity(result):
    return (
        1.0 - result["majority_correct"],
        1.0 - result["gold_support"],
        1.0 - result["conviction_score"],
        1.0 - result["reasoning_score"],
    )


def choose_minibatch(parent_results, minibatch_size):
    bad = []
    good = []

    for result in parent_results:
        if (
            result["majority_correct"] < 1.0
            or result["gold_support"] < 1.0
            or result["conviction_score"] < 1.0
            or result["reasoning_score"] < 1.0
        ):
            bad.append(result)
        else:
            good.append(result)

    bad.sort(key=example_severity, reverse=True)

    chosen = bad[: max(0, minibatch_size - 1)]

    if good:
        chosen.append(random.choice(good))

    if len(chosen) < minibatch_size:
        seen = {result["example_hash"] for result in chosen}
        for result in bad + good:
            if result["example_hash"] in seen:
                continue
            chosen.append(result)
            seen.add(result["example_hash"])
            if len(chosen) == minibatch_size:
                break

    seen_labels = {result["true_answer"] for result in chosen}
    if len(ORDERED_LABELS) == 2 and len(chosen) == minibatch_size:
        for label in ORDERED_LABELS:
            if label in seen_labels:
                continue
            replacement = next(
                (result for result in bad + good if result["true_answer"] == label and result not in chosen),
                None,
            )
            if replacement is not None:
                chosen[-1] = replacement

    return chosen


def select_reflection_examples(minibatch_results):
    return [result for result in minibatch_results if result["include_in_reflection"]] or minibatch_results


def sample_summary_lines(result):
    lines = [
        f"label_counts: {json.dumps(result['label_counts'], ensure_ascii=False)}",
        f"gold_support: {round(result['gold_support'], 3)}",
        f"conviction_score: {round(result['conviction_score'], 3)}",
        "conviction_score interpretation: 1.0 means all samples clustered on one label; 0.0 means maximum ordinal ambiguity.",
        f"majority_correct: {result['majority_correct']}",
        f"reasoning_score: {result['reasoning_score']}",
    ]

    for label in ORDERED_LABELS:
        reasonings = representative_reasonings(result["samples"], label)
        if reasonings:
            lines.append(f'{label.upper()}_EXAMPLES: {json.dumps(reasonings, ensure_ascii=False)}')

    return "\n".join(lines)


def build_mutation_prompt(parent_record, minibatch_results):
    parts = [
        "CURRENT SYSTEM PROMPT:",
        parent_record["candidate"]["system_prompt"],
        "",
        "CURRENT USER TEMPLATE:",
        parent_record["candidate"]["user_template"],
        "",
        "IMMUTABLE RESPONSE FORMAT SCHEMA:",
        json.dumps(TaskOutput.model_json_schema(), indent=2),
        "",
        "CURRENT FULL-DATASET METRICS:",
        json.dumps(
            {
                "precision": parent_record["precision"],
                "recall": parent_record["recall"],
                "f1": parent_record["f1"],
                "gold_support_perf": parent_record["gold_support_perf"],
                "conviction_perf": parent_record["conviction_perf"],
                "reasoning_perf": parent_record["reasoning_perf"],
            },
            indent=2,
        ),
        "",
        "CONVICTION SCORE INTERPRETATION:",
        "conviction_score ranges from 0.0 to 1.0.",
        "A score near 1.0 means the sampled outputs clustered on one label.",
        "A score near 0.0 means the sampled outputs were highly split across the ordered label space.",
        "Higher conviction is good only when the clustered label is also correct.",
        "",
        "LESSONS FROM EARLIER ACCEPTED MUTATIONS:",
    ]

    if parent_record["lesson_history"]:
        for i, lesson in enumerate(parent_record["lesson_history"][-5:], 1):
            parts.append(f"{i}. {lesson}")
    else:
        parts.append("None yet.")

    parts.append("")
    parts.append("MINIBATCH EXAMPLES:")

    for i, result in enumerate(minibatch_results, 1):
        parts.append(
            f"""
EXAMPLE {i}
TEXT:
{result["text"]}

GROUND TRUTH:
answer: {result["true_answer"]}
explanation: {result["true_explanation"]}

MAJORITY MODEL OUTPUT:
answer: {result["pred_answer"]}
reasoning: {result["pred_reasoning"]}

STABILITY SUMMARY:
{sample_summary_lines(result)}

DIAGNOSIS:
{result["diagnosis"]}

SUGGESTED FIX:
{result["suggested_fix"]}
""".strip()
        )
        parts.append("")

    return "\n".join(parts)


def mutate_candidate(parent_record, minibatch_results, mutation_prompt=None):
    if mutation_prompt is None:
        mutation_prompt = build_mutation_prompt(parent_record, minibatch_results)

    messages = [
        {
            "role": "system",
            "content": """
You are improving a prompt candidate for a structured extraction task.

Your goals:
- improve precision and recall, with F1 as the main optimization target
- preserve what already works
- reduce ambiguity when the same text yields mixed sampled labels
- increase conviction only when it supports the correct answer

Rules:
- Do NOT change the response format.
- The user_template must contain {{TEXT}} exactly once.
- Keep the prompt simple, explicit, and portable across models.
- conviction_score ranges from 0.0 to 1.0.
- A score near 1.0 means sampled outputs clustered on one label.
- A score near 0.0 means sampled outputs were highly split across the ordered label space.
- High conviction is desirable only when the clustered label is correct.
- Return only an improved system_prompt and user_template.
""".strip(),
        },
        {
            "role": "user",
            "content": mutation_prompt,
        },
    ]

    out = chat_parse_choices(
        model=MUTATE_MODEL,
        messages=messages,
        response_format=MutationOutput,
        schema_name="mutation_result",
        temperature=MUTATE_TEMPERATURE,
        n=1,
    )[0].model_dump()

    system_prompt = out["system_prompt"].strip() or parent_record["candidate"]["system_prompt"]
    user_template = out["user_template"].strip()

    if user_template.count("{{TEXT}}") != 1:
        user_template = parent_record["candidate"]["user_template"]

    return {
        "candidate": {
            "system_prompt": system_prompt,
            "user_template": user_template,
        },
        "diagnosis_summary": out["diagnosis_summary"].strip(),
    }


def build_record(candidate, full_eval, lesson_history):
    return {
        "candidate_hash": candidate_hash(candidate),
        "candidate": candidate,
        "full_results": full_eval["results"],
        "majority_accuracy": full_eval["majority_accuracy"],
        "gold_support_perf": full_eval["gold_support_perf"],
        "conviction_perf": full_eval["conviction_perf"],
        "reasoning_perf": full_eval["reasoning_perf"],
        "precision": full_eval["precision"],
        "recall": full_eval["recall"],
        "f1": full_eval["f1"],
        "lesson_history": lesson_history,
    }


def save_record(path, step, record):
    row = {"step": step, **record}
    with open(path, "a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_records(path):
    if not os.path.exists(path):
        return []

    records = []
    seen_hashes = set()

    with open(path, "r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            record_hash = row.get("candidate_hash") or candidate_hash(row["candidate"])
            if record_hash in seen_hashes:
                continue

            seen_hashes.add(record_hash)
            records.append(
                {
                    "candidate_hash": record_hash,
                    "candidate": row["candidate"],
                    "full_results": row["full_results"],
                    "majority_accuracy": row["majority_accuracy"],
                    "gold_support_perf": row["gold_support_perf"],
                    "conviction_perf": row["conviction_perf"],
                    "reasoning_perf": row["reasoning_perf"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "lesson_history": row.get("lesson_history", []),
                }
            )

    return records


def record_matches_dataset(record, dataset):
    if len(record["full_results"]) != len(dataset):
        return False

    for result, example in zip(record["full_results"], dataset):
        if result["text"] != example["text"]:
            return False
        if result["true_answer"] != example["true_answer"]:
            return False
        if result["true_explanation"] != example["true_explanation"]:
            return False

    return True


def build_frontier(records, dataset):
    frontier = {}

    for i in range(len(dataset)):
        best_score = max(
            (
                record["full_results"][i]["majority_correct"],
                record["full_results"][i]["gold_support"],
                record["full_results"][i]["reasoning_score"],
                record["full_results"][i]["conviction_score"],
            )
            for record in records
        )
        frontier[i] = [
            record
            for record in records
            if (
                record["full_results"][i]["majority_correct"],
                record["full_results"][i]["gold_support"],
                record["full_results"][i]["reasoning_score"],
                record["full_results"][i]["conviction_score"],
            )
            == best_score
        ]

    return frontier


def pick_parent(frontier):
    weighted_records = []
    for winners in frontier.values():
        weighted_records.extend(winners)
    return random.choice(weighted_records)


def frontier_size(frontier):
    hashes = set()
    for winners in frontier.values():
        for record in winners:
            hashes.add(record["candidate_hash"])
    return len(hashes)


def best_record(records):
    return max(records, key=summary_sort_key)


def optimize(initial_candidate, dataset, iterations=8, minibatch_size=MINIBATCH_SIZE, seed=0):
    random.seed(seed)

    archive = load_records(LOG_PATH)
    archive = [record for record in archive if record_matches_dataset(record, dataset)]

    if not archive:
        seed_eval = evaluate_full_dataset(initial_candidate, dataset, log_prefix="seed_full")
        seed_record = build_record(initial_candidate, seed_eval, lesson_history=[])
        archive.append(seed_record)
        save_record(LOG_PATH, 0, seed_record)

    best = best_record(archive)

    for step in range(1, iterations + 1):
        frontier = build_frontier(archive, dataset)
        known_hashes = {record["candidate_hash"] for record in archive}

        parent_record = pick_parent(frontier)
        parent_full_eval = summarize_results([dict(result) for result in parent_record["full_results"]])

        minibatch_results = choose_minibatch(parent_full_eval["results"], minibatch_size)
        reflection_examples = select_reflection_examples(minibatch_results)
        mutation_prompt = build_mutation_prompt(parent_record, reflection_examples)

        print_step_banner(step)
        print(f"PARENT HASH: {parent_record['candidate_hash']}")
        print_metric_block("PARENT FULL METRICS", parent_record)
        print_metric_block("PARENT MINIBATCH METRICS", summarize_results(minibatch_results))
        print("MINIBATCH EXAMPLES SELECTED FOR REFLECTION:")
        for result in reflection_examples:
            log_eval_result(result, prefix=f"step{step}_parent")
        print("MUTATION INPUT:")
        print(mutation_prompt)

        mutation = mutate_candidate(parent_record, reflection_examples, mutation_prompt=mutation_prompt)
        child_candidate = mutation["candidate"]
        child_hash = candidate_hash(child_candidate)

        if child_hash in known_hashes:
            print("Duplicate child candidate. Skipping.")
            continue

        child_minibatch_results = []
        child_minibatch_by_hash = {}

        for parent_result in minibatch_results:
            example = {
                "text": parent_result["text"],
                "true_answer": parent_result["true_answer"],
                "true_explanation": parent_result["true_explanation"],
            }
            result = evaluate_example(child_candidate, example, log_prefix=f"step{step}_mini")
            child_minibatch_results.append(result)
            child_minibatch_by_hash[result["example_hash"]] = result

        parent_minibatch_eval = summarize_results(minibatch_results)
        child_minibatch_eval = summarize_results(child_minibatch_results)

        print_metric_block("CHILD MINIBATCH METRICS", child_minibatch_eval)

        if summary_sort_key(child_minibatch_eval) <= summary_sort_key(parent_minibatch_eval):
            print("Child failed the minibatch acceptance test.")
            continue

        child_full_eval = evaluate_full_dataset(
            child_candidate,
            dataset,
            cached_results_by_example=child_minibatch_by_hash,
            log_prefix=f"step{step}_full",
        )

        child_lessons = list(parent_record["lesson_history"])
        if mutation["diagnosis_summary"]:
            child_lessons.append(mutation["diagnosis_summary"])

        child_record = build_record(child_candidate, child_full_eval, child_lessons)
        archive.append(child_record)
        save_record(LOG_PATH, len(archive) - 1, child_record)

        if summary_sort_key(child_record) > summary_sort_key(best):
            best = child_record

        frontier = build_frontier(archive, dataset)

        print_metric_block("CHILD FULL METRICS", child_record)
        print(f"FRONTIER CANDIDATES: {frontier_size(frontier)}")

    return {
        "best_candidate": best["candidate"],
        "best_candidate_hash": best["candidate_hash"],
        "best_f1": best["f1"],
        "best_recall": best["recall"],
        "best_precision": best["precision"],
        "best_support_perf": best["gold_support_perf"],
        "best_conviction_perf": best["conviction_perf"],
        "best_reasoning_perf": best["reasoning_perf"],
        "archive": archive,
    }


if __name__ == "__main__":
    dataset = [
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

    out = optimize(
        initial_candidate=make_initial_candidate(),
        dataset=dataset,
        iterations=2,
        minibatch_size=MINIBATCH_SIZE,
        seed=42,
    )

    print("\n" + "#" * 80)
    print("BEST HASH:\n")
    print(out["best_candidate_hash"])
    print("\nBEST SYSTEM PROMPT:\n")
    print(out["best_candidate"]["system_prompt"])
    print("\nBEST USER TEMPLATE:\n")
    print(out["best_candidate"]["user_template"])
    print("\nBEST F1       :", round(out["best_f1"], 3))
    print("BEST RECALL   :", round(out["best_recall"], 3))
    print("BEST PRECISION:", round(out["best_precision"], 3))
    print("BEST SUPPORT  :", round(out["best_support_perf"], 3))
    print("BEST CONVICT. :", round(out["best_conviction_perf"], 3))
    print("BEST REASON   :", round(out["best_reasoning_perf"], 3))

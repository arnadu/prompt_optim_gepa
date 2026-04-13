import hashlib
import json
import os
import random
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TASK_MODEL = "claude-haiku-4-5"
JUDGE_MODEL = "claude-haiku-4-5"
MUTATE_MODEL = "claude-haiku-4-5"

LOG_PATH = BASE_DIR / "gepa_lite_runs.jsonl"
MINIBATCH_SIZE = 4

_client = None


def get_client():
    global _client

    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is missing. Add it to prompt_optim_gepa/.env before running."
            )
        _client = Anthropic(api_key=api_key)

    return _client


IMMUTABLE_RESPONSE_FORMAT = {
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


def make_judge_response_format():
    return {
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
                    "include_in_reflection": {"type": "boolean"},
                },
                "required": [
                    "reasoning_score",
                    "diagnosis",
                    "suggested_fix",
                    "include_in_reflection",
                ],
                "additionalProperties": False,
            },
        },
    }


def make_mutation_response_format():
    return {
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


def response_format_instructions(response_format):
    schema = response_format["json_schema"]["schema"]
    return (
        "Return JSON only. Do not wrap it in markdown.\n"
        "Follow this JSON schema exactly:\n"
        f"{json.dumps(schema, indent=2)}"
    )


def extract_text_from_anthropic_response(response):
    parts = []

    for block in response.content:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    return "\n".join(parts).strip()


def extract_json_from_text(text):
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


def chat_json(model, messages, response_format):
    system_parts = []
    anthropic_messages = []

    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            anthropic_messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                }
            )

    system_parts.append(response_format_instructions(response_format))

    completion = get_client().messages.create(
        model=model,
        system="\n\n".join(system_parts),
        messages=anthropic_messages,
        max_tokens=1200,
    )

    text = extract_text_from_anthropic_response(completion)
    return json.loads(extract_json_from_text(text))


def make_initial_candidate():
    return {
        "system_prompt": """
You are an information extraction system for contracts.
Return a binary answer and a brief justification grounded in the text.
""".strip(),
        "user_template": "TEXT:\n{{TEXT}}",
    }

#Task:
#- Decide whether the text contains an automatic renewal clause.
#- Answer "Yes" when the agreement renews automatically unless someone opts out or terminates.
#- Answer "No" when renewal requires a new order, signature, amendment, or any other affirmative step.


def candidate_hash(candidate):
    payload = {
        "system_prompt": candidate["system_prompt"],
        "user_template": candidate["user_template"],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def example_hash(example):
    payload = {
        "text": example["text"],
        "true_answer": example["true_answer"],
        "true_explanation": example["true_explanation"],
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_user_message(user_template, text):
    return user_template.replace("{{TEXT}}", text)


def predict(candidate, text):
    messages = [
        {"role": "system", "content": candidate["system_prompt"]},
        {"role": "user", "content": render_user_message(candidate["user_template"], text)},
    ]

    out = chat_json(
        model=TASK_MODEL,
        messages=messages,
        response_format=IMMUTABLE_RESPONSE_FORMAT,
    )

    answer = out.get("answer", "No")
    if answer not in ["Yes", "No"]:
        answer = "No"

    return {
        "answer": answer,
        "reasoning": out.get("reasoning", ""),
    }


def judge(example, pred):
    messages = [
        {
            "role": "system",
            "content": """
You are evaluating an extraction result whose task is
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
answer: {pred["answer"]}
reasoning: {pred["reasoning"]}
""".strip(),
        },
    ]

    out = chat_json(
        model=JUDGE_MODEL,
        messages=messages,
        response_format=make_judge_response_format(),
    )

    if pred["answer"] != example["true_answer"]:
        out["reasoning_score"] = 0

    return out


def evaluate_example(candidate, example):
    pred = predict(candidate, example["text"])
    judge_out = judge(example, pred)

    return {
        "example_hash": example_hash(example),
        "text": example["text"],
        "true_answer": example["true_answer"],
        "true_explanation": example["true_explanation"],
        "pred_answer": pred["answer"],
        "pred_reasoning": pred["reasoning"],
        "answer_score": 1.0 if pred["answer"] == example["true_answer"] else 0.0,
        "reasoning_score": float(judge_out["reasoning_score"]) / 2.0,
        "diagnosis": judge_out["diagnosis"],
        "suggested_fix": judge_out["suggested_fix"],
        "include_in_reflection": bool(judge_out["include_in_reflection"]),
    }


def summarize_results(results):
    answer_total = sum(result["answer_score"] for result in results)
    reasoning_total = sum(result["reasoning_score"] for result in results)

    return {
        "results": results,
        "answer_total": answer_total,
        "reasoning_total": reasoning_total,
        "answer_perf": answer_total / len(results),
        "reasoning_perf": reasoning_total / len(results),
    }


def evaluate_full_dataset(candidate, dataset, cached_results_by_example=None):
    cached_results_by_example = cached_results_by_example or {}
    results = []

    for example in dataset:
        ex_hash = example_hash(example)
        if ex_hash in cached_results_by_example:
            results.append(dict(cached_results_by_example[ex_hash]))
        else:
            results.append(evaluate_example(candidate, example))

    return summarize_results(results)


def choose_minibatch(parent_results, minibatch_size):
    bad = []
    good = []

    for result in parent_results:
        score = result["answer_score"] + result["reasoning_score"]
        if result["answer_score"] < 1.0 or result["reasoning_score"] < 1.0:
            bad.append((score, result))
        else:
            good.append(result)

    bad.sort(key=lambda item: item[0])

    chosen = [result for _, result in bad[: max(0, minibatch_size - 1)]]

    if good:
        chosen.append(random.choice(good))

    if len(chosen) < minibatch_size:
        all_results = sorted(
            parent_results,
            key=lambda result: (result["answer_score"] + result["reasoning_score"]),
        )
        seen = {result["example_hash"] for result in chosen}
        for result in all_results:
            if result["example_hash"] in seen:
                continue
            chosen.append(result)
            seen.add(result["example_hash"])
            if len(chosen) == minibatch_size:
                break

    return chosen


def select_reflection_examples(minibatch_results):
    return minibatch_results


def build_mutation_prompt(parent_record, minibatch_results):
    parts = []

    parts.append("CURRENT SYSTEM PROMPT:")
    parts.append(parent_record["candidate"]["system_prompt"])
    parts.append("")

    parts.append("CURRENT USER TEMPLATE:")
    parts.append(parent_record["candidate"]["user_template"])
    parts.append("")

    parts.append("IMMUTABLE RESPONSE FORMAT:")
    parts.append(json.dumps(IMMUTABLE_RESPONSE_FORMAT, indent=2))
    parts.append("")

    parts.append("LESSONS FROM EARLIER ACCEPTED MUTATIONS:")
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

MODEL OUTPUT:
answer: {result["pred_answer"]}
reasoning: {result["pred_reasoning"]}

SCORES:
answer_score: {result["answer_score"]}
reasoning_score: {result["reasoning_score"]}

DIAGNOSIS:
{result["diagnosis"]}

SUGGESTED FIX:
{result["suggested_fix"]}
""".strip()
        )
        parts.append("")

    return "\n".join(parts)


def mutate_candidate(parent_record, minibatch_results):
    messages = [
        {
            "role": "system",
            "content": """
You are improving a prompt candidate for a structured extraction task.

Your goals:
- fix the failures shown in the minibatch
- preserve what already works
- keep the prompt simple and explicit

Rules:
- Do NOT change the response format.
- The user_template must contain {{TEXT}} exactly once.
- Return only an improved system_prompt and user_template.
""".strip(),
        },
        {
            "role": "user",
            "content": build_mutation_prompt(parent_record, minibatch_results),
        },
    ]

    out = chat_json(
        model=MUTATE_MODEL,
        messages=messages,
        response_format=make_mutation_response_format(),
    )

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
        "answer_perf": full_eval["answer_perf"],
        "reasoning_perf": full_eval["reasoning_perf"],
        "lesson_history": lesson_history,
    }


def save_record(path, step, record):
    row = {
        "step": step,
        "candidate_hash": record["candidate_hash"],
        "candidate": record["candidate"],
        "full_results": record["full_results"],
        "answer_perf": record["answer_perf"],
        "reasoning_perf": record["reasoning_perf"],
        "lesson_history": record["lesson_history"],
    }

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
                    "answer_perf": row["answer_perf"],
                    "reasoning_perf": row["reasoning_perf"],
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
        best_score = max(record["full_results"][i]["answer_score"] for record in records)
        winners = []

        for record in records:
            if record["full_results"][i]["answer_score"] == best_score:
                winners.append(record)

        frontier[i] = winners

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
    best = records[0]

    for record in records[1:]:
        if (
            record["answer_perf"] > best["answer_perf"]
            or (
                record["answer_perf"] == best["answer_perf"]
                and record["reasoning_perf"] > best["reasoning_perf"]
            )
        ):
            best = record

    return best


def optimize(initial_candidate, dataset, iterations=8, minibatch_size=MINIBATCH_SIZE, seed=0):
    random.seed(seed)

    archive = load_records(LOG_PATH)
    archive = [record for record in archive if record_matches_dataset(record, dataset)]

    if not archive:
        seed_eval = evaluate_full_dataset(initial_candidate, dataset)
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
        mutation = mutate_candidate(parent_record, reflection_examples)
        child_candidate = mutation["candidate"]
        child_hash = candidate_hash(child_candidate)

        if child_hash in known_hashes:
            print("\n" + "=" * 80)
            print(f"STEP {step}")
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
            result = evaluate_example(child_candidate, example)
            child_minibatch_results.append(result)
            child_minibatch_by_hash[result["example_hash"]] = result

        parent_minibatch_eval = summarize_results(minibatch_results)
        child_minibatch_eval = summarize_results(child_minibatch_results)

        if child_minibatch_eval["answer_total"] <= parent_minibatch_eval["answer_total"]:
            print("\n" + "=" * 80)
            print(f"STEP {step}")
            print("Child failed the minibatch acceptance test.")
            print("PARENT MINIBATCH ANSWER:", round(parent_minibatch_eval["answer_perf"], 3))
            print("CHILD MINIBATCH ANSWER :", round(child_minibatch_eval["answer_perf"], 3))
            continue

        child_full_eval = evaluate_full_dataset(
            child_candidate,
            dataset,
            cached_results_by_example=child_minibatch_by_hash,
        )

        child_lessons = list(parent_record["lesson_history"])
        if mutation["diagnosis_summary"]:
            child_lessons.append(mutation["diagnosis_summary"])

        child_record = build_record(child_candidate, child_full_eval, child_lessons)
        archive.append(child_record)
        save_record(LOG_PATH, len(archive) - 1, child_record)

        if (
            child_record["answer_perf"] > best["answer_perf"]
            or (
                child_record["answer_perf"] == best["answer_perf"]
                and child_record["reasoning_perf"] > best["reasoning_perf"]
            )
        ):
            best = child_record

        frontier = build_frontier(archive, dataset)

        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("PARENT FULL ANSWER     :", round(parent_record["answer_perf"], 3))
        print("PARENT MINIBATCH ANSWER:", round(parent_minibatch_eval["answer_perf"], 3))
        print("CHILD MINIBATCH ANSWER :", round(child_minibatch_eval["answer_perf"], 3))
        print("CHILD FULL ANSWER      :", round(child_record["answer_perf"], 3))
        print("CHILD FULL REASON      :", round(child_record["reasoning_perf"], 3))
        print("FRONTIER CANDIDATES    :", frontier_size(frontier))

    return {
        "best_candidate": best["candidate"],
        "best_candidate_hash": best["candidate_hash"],
        "best_answer_perf": best["answer_perf"],
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
    print("\nBEST FULL ANSWER PERF:", round(out["best_answer_perf"], 3))
    print("BEST FULL REASON PERF:", round(out["best_reasoning_perf"], 3))

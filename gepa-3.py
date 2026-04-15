"""
GEPA v3 – Gradient-free Evolutionary Prompt Adaptation with conviction sampling.

Overview
--------
Each candidate prompt is evaluated by sampling the task LLM k times per example.
The spread of those k answers is measured with Leik's D (an ordinal dispersion
measure). Conviction = 1 − Leik's D: a prompt that reliably produces the same
correct answer scores near 1.0; one that flips unpredictably scores near 0.0.

A judge LLM diagnoses two distinct failure modes for each example:
  - correctness failure: the majority answer is wrong
  - consistency failure: the samples flip between labels (low conviction)

A mutator LLM receives these diagnoses, grouped by failure type, and produces
an improved candidate prompt. GEPA maintains a Pareto frontier and samples parents
proportional to how many examples they dominate.

Persistence
-----------
All LLM calls are cached in LLM_CACHE_PATH. The cache key includes the model,
messages, schema, n, and temperature, so schema changes auto-invalidate old entries.
Interrupted runs resume without any repeated API calls.
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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

TASK_MODEL   = os.environ.get("OPENAI_TASK_MODEL",   "gpt-4o-mini")
JUDGE_MODEL  = os.environ.get("OPENAI_JUDGE_MODEL",  "gpt-4o-mini")
MUTATE_MODEL = os.environ.get("OPENAI_MUTATE_MODEL", "gpt-4o-mini")

TASK_SAMPLES        = 5
TASK_TEMPERATURE    = 1.0   # high temperature encourages diverse samples
JUDGE_TEMPERATURE   = 0.2
MUTATE_TEMPERATURE  = 0.4

MINIBATCH_SIZE = 4

LLM_CACHE_PATH = BASE_DIR / "gepa3_llm_cache.jsonl"
RUNS_PATH      = BASE_DIR / "gepa3_runs.jsonl"

ORDERED_LABELS = ["No", "Yes"]   # ordered low → high for Leik's D
POSITIVE_LABEL = "Yes"


# ---------------------------------------------------------------------------
# Pydantic output schemas
# ---------------------------------------------------------------------------

class TaskOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: Literal["No", "Yes"]
    reasoning: str


class JudgeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reasoning_score: Literal[0, 1, 2]        # 0=wrong/weak  1=partial  2=correct+specific
    correctness_analysis: str                 # is the majority answer right, and why?
    consistency_analysis: str                 # why do samples flip labels? (or "Consistent." if they don't)
    suggested_fix: str                        # one concrete prompt-level improvement
    include_in_reflection: bool


class MutationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    diagnosis_summary: str
    system_prompt: str
    user_template: str


# ---------------------------------------------------------------------------
# OpenAI client + cached call_llm
# ---------------------------------------------------------------------------

_client: OpenAI | None = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set. Add it to prompt_optim_gepa/.env.")
        _client = OpenAI(api_key=api_key)
    return _client


_llm_cache: dict | None = None

def call_llm(model, messages, schema, schema_name, temperature, n=1) -> list[BaseModel]:
    """
    Call the LLM and return n parsed Pydantic objects.

    All calls are cached in LLM_CACHE_PATH. The cache key covers the full call
    signature — including the schema — so changing any output schema field
    automatically invalidates its old cache entries.
    The raw JSON response strings are preserved in the cache, so all fields
    (including reasoning) are available when replaying from disk.
    """
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = {}
        if LLM_CACHE_PATH.exists():
            for line in LLM_CACHE_PATH.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    _llm_cache[entry["key"]] = entry["responses"]

    key = make_hash({"model": model, "messages": messages,
                     "schema": schema.model_json_schema(), "n": n, "temperature": temperature})

    if key not in _llm_cache:
        completion = get_client().chat.completions.create(
            model=model, messages=messages, n=n, temperature=temperature,
            max_completion_tokens=1200,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema.model_json_schema()},
            },
        )
        _llm_cache[key] = [c.message.content or "" for c in completion.choices]
        with open(LLM_CACHE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "responses": _llm_cache[key]}) + "\n")

    return [schema.model_validate_json(r) for r in _llm_cache[key]]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def make_hash(obj: dict) -> str:
    canonical = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def leik_d(counts: dict[str, int]) -> float:
    """
    Ordinal dispersion over ORDERED_LABELS.
    D = 0 when all samples agree; D = 1 when maximally spread.

    Formula: D = 2/(K-1) * sum_{k=1}^{K-1} min(F_k, 1 - F_k)
    where F_k is the cumulative proportion up to the k-th category.
    """
    total = sum(counts.values())
    if total == 0 or len(ORDERED_LABELS) <= 1:
        return 0.0
    cumulative = dispersion_sum = 0.0
    for label in ORDERED_LABELS[:-1]:
        cumulative += counts[label] / total
        dispersion_sum += min(cumulative, 1.0 - cumulative)
    return (2.0 * dispersion_sum) / (len(ORDERED_LABELS) - 1)


def sample_reasonings(samples: list[dict], label: str, limit: int = 2) -> list[str]:
    """Return up to `limit` non-empty reasoning strings from samples that chose `label`."""
    return [s["reasoning"] for s in samples if s["answer"] == label and s["reasoning"]][:limit]


# ---------------------------------------------------------------------------
# Predict: sample the task LLM k times for one example
# ---------------------------------------------------------------------------

def predict(candidate: dict, text: str) -> dict:
    """
    Run the candidate prompt on `text` TASK_SAMPLES times (one batched API call).
    Returns all k samples plus derived statistics: label counts, majority vote, conviction.
    """
    messages = [
        {"role": "system", "content": candidate["system_prompt"]},
        {"role": "user",   "content": candidate["user_template"].replace("{{TEXT}}", text)},
    ]
    parsed  = call_llm(TASK_MODEL, messages, TaskOutput, "extraction_result", TASK_TEMPERATURE, n=TASK_SAMPLES)
    samples = [{"answer": r.answer, "reasoning": r.reasoning} for r in parsed]

    counts = {label: sum(1 for s in samples if s["answer"] == label) for label in ORDERED_LABELS}
    predicted_label = max(ORDERED_LABELS, key=lambda lbl: (counts[lbl], -ORDERED_LABELS.index(lbl)))

    return {
        "samples":             samples,
        "counts":              counts,
        "distribution":        {lbl: counts[lbl] / TASK_SAMPLES for lbl in ORDERED_LABELS},
        "predicted_label":     predicted_label,
        "predicted_reasoning": next((s["reasoning"] for s in samples if s["answer"] == predicted_label), ""),
        "conviction":          1.0 - leik_d(counts),
    }


# ---------------------------------------------------------------------------
# Judge: diagnose correctness and consistency failures
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """
You are evaluating a prompt candidate for binary contract clause extraction.

Task: decide whether a contract text contains an automatic renewal clause.
  "Yes" = renews automatically unless someone opts out.
  "No"  = renewal requires an affirmative step (new signature, purchase order, etc.)

The model was sampled k times (conviction = 1 − Leik's D):
  1.0 = all k samples gave the same answer (fully consistent)
  0.0 = samples were maximally split (fully inconsistent)

Provide:

correctness_analysis — Was the majority answer correct?
  If wrong: identify the specific reasoning error and what the prompt failed to clarify.
  If correct: briefly confirm why.

consistency_analysis — Did the samples flip between labels?
  If conviction < 1.0: explain what is ambiguous or under-specified in the prompt
  that causes the model to flip-flop on this text. Quote specific phrases that trigger ambiguity.
  If conviction = 1.0: write "Consistent."

suggested_fix — One concrete prompt-level improvement that generalises across documents.
  Do NOT suggest document-specific facts.

reasoning_score:
  2 = correct answer with specific, well-grounded reasoning
  1 = correct answer but vague or incomplete reasoning
  0 = wrong answer (always 0 if majority answer is wrong)

include_in_reflection — true if this example would help improve the prompt.
""".strip()


def judge(example: dict, prediction: dict) -> dict:
    """Ask the judge LLM to diagnose correctness and consistency failures."""
    yes_reasonings = sample_reasonings(prediction["samples"], "Yes")
    no_reasonings  = sample_reasonings(prediction["samples"], "No")

    user_content = f"""
TEXT:
{example["text"]}

GROUND TRUTH: {example["true_answer"]}
EXPLANATION:  {example["true_explanation"]}

SAMPLED COUNTS:     {json.dumps(prediction["counts"])}
CONVICTION:         {round(prediction["conviction"], 3)}

MAJORITY ANSWER:    {prediction["predicted_label"]}
MAJORITY REASONING: {prediction["predicted_reasoning"]}

ALL "YES" REASONINGS: {json.dumps(yes_reasonings, ensure_ascii=False)}
ALL "NO"  REASONINGS: {json.dumps(no_reasonings,  ensure_ascii=False)}
""".strip()

    out = call_llm(
        JUDGE_MODEL,
        [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}, {"role": "user", "content": user_content}],
        JudgeOutput, "judge_result", JUDGE_TEMPERATURE,
    )[0].model_dump()

    if prediction["predicted_label"] != example["true_answer"]:
        out["reasoning_score"] = 0   # wrong answer → reasoning quality is irrelevant

    return out


# ---------------------------------------------------------------------------
# Evaluate a single example (purely functional — caching is inside call_llm)
# ---------------------------------------------------------------------------

def evaluate_example(candidate: dict, example: dict) -> dict:
    """
    Run predict + judge for one (candidate, example) pair and return a result dict.
    LLM calls are transparently cached inside call_llm, so this function is pure
    from the caller's perspective.
    """
    try:
        prediction = predict(candidate, example["text"])
        judge_out  = judge(example, prediction)
    except Exception as exc:
        prediction = {
            "samples": [], "counts": {lbl: 0 for lbl in ORDERED_LABELS},
            "distribution": {lbl: 0.0 for lbl in ORDERED_LABELS},
            "predicted_label": ORDERED_LABELS[0], "predicted_reasoning": "", "conviction": 0.0,
        }
        judge_out = {
            "reasoning_score": 0,
            "correctness_analysis": f"Evaluation failed: {type(exc).__name__}: {exc}",
            "consistency_analysis": "",
            "suggested_fix": "Make the prompt and output instructions clearer and more explicit.",
            "include_in_reflection": True,
        }

    n_samples           = max(1, sum(prediction["counts"].values()))
    true_label_fraction = prediction["counts"][example["true_answer"]] / n_samples

    return {
        "example_hash":           make_hash(example),
        "text":                   example["text"],
        "true_answer":            example["true_answer"],
        "true_explanation":       example["true_explanation"],
        "samples":                prediction["samples"],
        "counts":                 prediction["counts"],
        "distribution":           prediction["distribution"],
        "predicted_label":        prediction["predicted_label"],
        "predicted_reasoning":    prediction["predicted_reasoning"],
        "correct":                float(prediction["predicted_label"] == example["true_answer"]),
        "true_label_fraction":    true_label_fraction,
        "conviction":             prediction["conviction"],
        "reasoning_score":        float(judge_out["reasoning_score"]) / 2.0,
        "correctness_analysis":   judge_out["correctness_analysis"],
        "consistency_analysis":   judge_out["consistency_analysis"],
        "suggested_fix":          judge_out["suggested_fix"],
        "include_in_reflection":  bool(judge_out["include_in_reflection"]),
    }


# ---------------------------------------------------------------------------
# Dataset-level metrics
# ---------------------------------------------------------------------------

def compute_dataset_metrics(results: list[dict]) -> dict:
    """
    Compute binary F1/precision/recall and per-example averages.
    Includes a `score` tuple (higher = better) used throughout for candidate comparison.
    """
    tp = fp = tn = fn = 0
    for r in results:
        pred_pos = r["predicted_label"] == POSITIVE_LABEL
        true_pos = r["true_answer"]     == POSITIVE_LABEL
        if   pred_pos and     true_pos: tp += 1
        elif pred_pos and not true_pos: fp += 1
        elif true_pos and not pred_pos: fn += 1
        else:                           tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n         = len(results)

    avg_correct             = sum(r["correct"]             for r in results) / n
    avg_true_label_fraction = sum(r["true_label_fraction"] for r in results) / n
    avg_conviction          = sum(r["conviction"]          for r in results) / n
    avg_reasoning_score     = sum(r["reasoning_score"]     for r in results) / n

    # Ranking tuple: higher is better. Priority: F1 > recall > precision > support > reasoning > conviction.
    score = (f1, recall, precision, avg_true_label_fraction, avg_reasoning_score, avg_conviction, avg_correct)

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "avg_correct":             avg_correct,
        "avg_true_label_fraction": avg_true_label_fraction,
        "avg_conviction":          avg_conviction,
        "avg_reasoning_score":     avg_reasoning_score,
        "score":                   score,
    }


def evaluate_dataset(candidate, dataset, log_prefix="") -> dict:
    """Evaluate `candidate` on every example in `dataset`."""
    results = []
    for example in dataset:
        result = evaluate_example(candidate, example)
        if log_prefix:
            print_eval_result(result, prefix=log_prefix)
        results.append(result)
    return {"results": results, **compute_dataset_metrics(results)}


# ---------------------------------------------------------------------------
# Minibatch selection
# ---------------------------------------------------------------------------

def example_badness(result: dict) -> tuple:
    """Higher = more problematic. Sort key for minibatch selection."""
    return (1.0 - result["correct"], 1.0 - result["true_label_fraction"],
            1.0 - result["conviction"], 1.0 - result["reasoning_score"])


def choose_minibatch(results: list[dict], minibatch_size: int) -> list[dict]:
    """
    Select a minibatch for the mutation step.
    - Most slots: worst-performing examples (sorted by badness).
    - One slot: a random perfect example so the mutator sees what works.
    - Ensure both labels are represented (binary tasks).
    """
    perfect = lambda r: r["correct"] == r["true_label_fraction"] == r["conviction"] == r["reasoning_score"] == 1.0
    bad  = sorted([r for r in results if not perfect(r)], key=example_badness, reverse=True)
    good = [r for r in results if perfect(r)]

    chosen = bad[: max(0, minibatch_size - 1)]
    if good:
        chosen.append(random.choice(good))

    if len(chosen) < minibatch_size:
        chosen_hashes = {r["example_hash"] for r in chosen}
        for r in bad:
            if r["example_hash"] not in chosen_hashes:
                chosen.append(r)
                chosen_hashes.add(r["example_hash"])
                if len(chosen) == minibatch_size:
                    break

    if len(ORDERED_LABELS) == 2 and len(chosen) == minibatch_size:
        chosen_hashes  = {r["example_hash"] for r in chosen}
        present_labels = {r["true_answer"] for r in chosen}
        for missing_label in ORDERED_LABELS:
            if missing_label in present_labels:
                continue
            replacement = next(
                (r for r in results if r["true_answer"] == missing_label and r["example_hash"] not in chosen_hashes), None
            )
            if replacement is not None:
                chosen[-1] = replacement
                chosen_hashes  = {r["example_hash"] for r in chosen}
                present_labels = {r["true_answer"] for r in chosen}

    return chosen


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def build_mutation_prompt(record: dict, minibatch: list[dict]) -> str:
    """
    Assemble the context block for the mutator LLM.

    Examples are grouped into three sections so the mutator can address each
    failure mode with a targeted fix:
      - Wrong answer (correctness failure, possibly also inconsistent)
      - Flip-flop (correct majority but low conviction — consistency failure only)
      - Perfect examples (what the prompt already handles well)
    """
    wrong     = [r for r in minibatch if r["correct"] < 1.0]
    flip_flop = [r for r in minibatch if r["correct"] == 1.0 and r["conviction"] < 1.0]
    perfect   = [r for r in minibatch if r["correct"] == 1.0 and r["conviction"] == 1.0]

    def example_block(result, idx):
        yes_r = sample_reasonings(result["samples"], "Yes")
        no_r  = sample_reasonings(result["samples"], "No")
        lines = [
            f"EXAMPLE {idx}",
            f"TEXT: {result['text']}",
            f"GROUND TRUTH: {result['true_answer']} — {result['true_explanation']}",
            f"MAJORITY OUTPUT: {result['predicted_label']} (conviction={round(result['conviction'], 2)}, counts={result['counts']})",
        ]
        if yes_r: lines.append(f"YES REASONINGS: {json.dumps(yes_r, ensure_ascii=False)}")
        if no_r:  lines.append(f"NO  REASONINGS: {json.dumps(no_r,  ensure_ascii=False)}")
        lines += [
            f"CORRECTNESS: {result['correctness_analysis']}",
            f"CONSISTENCY: {result['consistency_analysis']}",
            f"SUGGESTED FIX: {result['suggested_fix']}",
        ]
        return "\n".join(lines)

    parts = [
        "CURRENT SYSTEM PROMPT:", record["candidate"]["system_prompt"], "",
        "CURRENT USER TEMPLATE:", record["candidate"]["user_template"], "",
        "IMMUTABLE RESPONSE FORMAT SCHEMA:", json.dumps(TaskOutput.model_json_schema(), indent=2), "",
        "FULL-DATASET METRICS:", json.dumps({
            k: record[k] for k in ("precision", "recall", "f1", "avg_true_label_fraction", "avg_conviction", "avg_reasoning_score")
        }, indent=2), "",
        "CONVICTION: 1.0 = all k samples agreed; 0.0 = maximally split. High conviction is only good when the label is also correct.", "",
        "LESSONS FROM EARLIER ACCEPTED MUTATIONS:",
    ]
    parts += [f"{i}. {l}" for i, l in enumerate(record["lesson_history"][-5:], 1)] or ["None yet."]

    if wrong:
        parts += ["", "── WRONG ANSWER EXAMPLES (fix the decision rule) ──"]
        parts += [example_block(r, i + 1) for i, r in enumerate(wrong)]
    if flip_flop:
        parts += ["", "── FLIP-FLOP EXAMPLES (fix ambiguity — majority is correct but inconsistent) ──"]
        parts += [example_block(r, i + 1) for i, r in enumerate(flip_flop)]
    if perfect:
        parts += ["", "── PERFECT EXAMPLES (preserve this behaviour) ──"]
        parts += [f"EXAMPLE {i + 1}: {r['true_answer']} — {r['text'][:120]}" for i, r in enumerate(perfect)]

    return "\n".join(parts)


MUTATOR_SYSTEM_PROMPT = """
You are improving a prompt candidate for a structured extraction task.

Goals:
- Improve precision and recall (F1 is the main optimisation target).
- Fix wrong-answer examples by sharpening the decision rule.
- Fix flip-flop examples by removing ambiguity so the model converges consistently.
- Preserve what already works (the perfect examples).
- Increase conviction only when it supports the correct answer.

Rules:
- Do NOT change the response format.
- The user_template must contain {{TEXT}} exactly once.
- Keep the prompt simple, explicit, and portable across models.
- Return only an improved system_prompt and user_template.
""".strip()


def mutate_candidate(record: dict, minibatch: list[dict]) -> dict:
    """Ask the mutator LLM to produce an improved candidate prompt."""
    messages = [
        {"role": "system", "content": MUTATOR_SYSTEM_PROMPT},
        {"role": "user",   "content": build_mutation_prompt(record, minibatch)},
    ]
    out = call_llm(MUTATE_MODEL, messages, MutationOutput, "mutation_result", MUTATE_TEMPERATURE)[0].model_dump()

    system_prompt = out["system_prompt"].strip() or record["candidate"]["system_prompt"]
    user_template = out["user_template"].strip()
    if user_template.count("{{TEXT}}") != 1:
        user_template = record["candidate"]["user_template"]

    return {
        "candidate":         {"system_prompt": system_prompt, "user_template": user_template},
        "diagnosis_summary": out["diagnosis_summary"].strip(),
    }


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def build_frontier(archive: list[dict], n_examples: int) -> dict[int, list[dict]]:
    """
    For each example index, find all archive records that achieve the best
    (correct, true_label_fraction, reasoning_score, conviction) score on it.

    Records that dominate more examples appear more often, giving them a higher
    probability of being picked as a parent.
    """
    def example_score(record, i):
        r = record["full_results"][i]
        return (r["correct"], r["true_label_fraction"], r["reasoning_score"], r["conviction"])

    frontier = {}
    for i in range(n_examples):
        best = max(example_score(r, i) for r in archive)
        frontier[i] = [r for r in archive if example_score(r, i) == best]
    return frontier


# ---------------------------------------------------------------------------
# Archive management
# ---------------------------------------------------------------------------

def load_archive(path: Path) -> list[dict]:
    """Load all unique candidate records from the runs log."""
    if not path.exists():
        return []
    archive, seen = [], set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row       = json.loads(line)
        cand_hash = row.get("candidate_hash") or make_hash(row["candidate"])
        if cand_hash in seen:
            continue
        seen.add(cand_hash)
        archive.append({
            "candidate_hash":          cand_hash,
            "candidate":               row["candidate"],
            "full_results":            row["full_results"],
            "f1":                      row["f1"],
            "recall":                  row["recall"],
            "precision":               row["precision"],
            "avg_correct":             row["avg_correct"],
            "avg_true_label_fraction": row["avg_true_label_fraction"],
            "avg_conviction":          row["avg_conviction"],
            "avg_reasoning_score":     row["avg_reasoning_score"],
            "lesson_history":          row.get("lesson_history", []),
            # score is a tuple; JSON round-trips it as a list, so reconstruct here
            "score": tuple(row["score"]) if "score" in row else (
                row["f1"], row["recall"], row["precision"],
                row["avg_true_label_fraction"], row["avg_reasoning_score"],
                row["avg_conviction"], row["avg_correct"],
            ),
        })
    return archive


def archive_matches_dataset(archive: list[dict], dataset: list[dict]) -> bool:
    """Return True only if every record's stored results align with the current dataset."""
    for record in archive:
        stored = record["full_results"]
        if len(stored) != len(dataset):
            return False
        if any(r["text"] != e["text"] or r["true_answer"] != e["true_answer"] or r["true_explanation"] != e["true_explanation"]
               for r, e in zip(stored, dataset)):
            return False
    return True


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_eval_result(result: dict, prefix: str = "EVAL"):
    n          = sum(result["counts"].values())
    counts_str = "/".join(f"{lbl}={result['counts'][lbl]}" for lbl in ORDERED_LABELS)
    preview    = " ".join(result["text"].split())
    preview    = preview if len(preview) <= 72 else preview[:69] + "..."
    if n == 0:
        print(f"[{prefix}] ERROR pred={result['predicted_label']} gold={result['true_answer']} "
              f"correctness=\"{result['correctness_analysis'][:100]}\"")
    else:
        print(f"[{prefix}] pred={result['predicted_label']} gold={result['true_answer']} "
              f"n={n} counts={counts_str} conviction={result['conviction']:.2f} "
              f"support={result['true_label_fraction']:.2f} reason={result['reasoning_score']:.2f} "
              f"text=\"{preview}\"")


def print_metrics(title: str, metrics: dict):
    print(title)
    print(f"  F1         : {metrics['f1']:.3f}")
    print(f"  Recall     : {metrics['recall']:.3f}")
    print(f"  Precision  : {metrics['precision']:.3f}")
    print(f"  Support    : {metrics['avg_true_label_fraction']:.3f}")
    print(f"  Conviction : {metrics['avg_conviction']:.3f}")
    print(f"  Reasoning  : {metrics['avg_reasoning_score']:.3f}")


# ---------------------------------------------------------------------------
# Main optimisation loop
# ---------------------------------------------------------------------------

def optimize(initial_candidate, dataset, iterations=8, minibatch_size=MINIBATCH_SIZE, seed=0) -> dict:
    """
    Run the GEPA evolutionary loop.

    Each iteration:
      1. Pick a parent from the Pareto frontier (proportional to examples dominated).
      2. Choose a minibatch of difficult examples from the parent's results.
      3. Mutate the parent prompt using the judge's diagnoses.
      4. Accept the child only if it improves the minibatch score.
      5. Evaluate the accepted child on the full dataset.
      6. Update the archive and frontier.
    """
    random.seed(seed)

    archive = load_archive(RUNS_PATH)
    if archive and not archive_matches_dataset(archive, dataset):
        print("WARNING: archived records do not match current dataset. Starting fresh.")
        archive = []

    if not archive:
        print("No archive found. Evaluating initial candidate on full dataset...")
        result      = evaluate_dataset(initial_candidate, dataset, log_prefix="seed")
        seed_record = {"candidate_hash": make_hash(initial_candidate), "candidate": initial_candidate,
                       "full_results": result["results"], "lesson_history": [],
                       **{k: result[k] for k in ("f1", "recall", "precision", "avg_correct",
                          "avg_true_label_fraction", "avg_conviction", "avg_reasoning_score", "score")}}
        archive.append(seed_record)
        with open(RUNS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": 0, **seed_record}, ensure_ascii=False) + "\n")

    best = max(archive, key=lambda r: r["score"])

    for step in range(1, iterations + 1):
        frontier     = build_frontier(archive, n_examples=len(dataset))
        known_hashes = {r["candidate_hash"] for r in archive}

        parent     = random.choice([r for winners in frontier.values() for r in winners])
        minibatch  = choose_minibatch(parent["full_results"], minibatch_size)
        reflection = [r for r in minibatch if r["include_in_reflection"]] or minibatch

        print("\n" + "=" * 80)
        print(f"STEP {step}  parent={parent['candidate_hash']}")
        print_metrics("PARENT FULL METRICS",      parent)
        print_metrics("PARENT MINIBATCH METRICS", compute_dataset_metrics(minibatch))
        print("MUTATION INPUT:")
        print(build_mutation_prompt(parent, reflection))

        mutation        = mutate_candidate(parent, reflection)
        child_candidate = mutation["candidate"]
        child_hash      = make_hash(child_candidate)

        if child_hash in known_hashes:
            print(f"Step {step}: duplicate candidate — skipping.")
            continue

        # Quick minibatch check before committing to a full-dataset eval
        child_minibatch_results = []
        for parent_result in minibatch:
            example = {k: parent_result[k] for k in ("text", "true_answer", "true_explanation")}
            result  = evaluate_example(child_candidate, example)
            print_eval_result(result, prefix=f"step{step}_mini")
            child_minibatch_results.append(result)

        child_mini_metrics  = compute_dataset_metrics(child_minibatch_results)
        parent_mini_metrics = compute_dataset_metrics(minibatch)

        print_metrics("CHILD MINIBATCH METRICS", child_mini_metrics)

        if child_mini_metrics["score"] <= parent_mini_metrics["score"]:
            print(f"Step {step}: child did not improve on the minibatch — skipping.")
            continue

        child_result = evaluate_dataset(child_candidate, dataset, log_prefix=f"step{step}_full")

        lessons = list(parent["lesson_history"])
        if mutation["diagnosis_summary"]:
            lessons.append(mutation["diagnosis_summary"])

        child_record = {"candidate_hash": child_hash, "candidate": child_candidate,
                        "full_results": child_result["results"], "lesson_history": lessons,
                        **{k: child_result[k] for k in ("f1", "recall", "precision", "avg_correct",
                           "avg_true_label_fraction", "avg_conviction", "avg_reasoning_score", "score")}}
        archive.append(child_record)
        with open(RUNS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": len(archive) - 1, **child_record}, ensure_ascii=False) + "\n")

        if child_record["score"] > best["score"]:
            best = child_record

        frontier = build_frontier(archive, n_examples=len(dataset))
        n_frontier = len({r["candidate_hash"] for winners in frontier.values() for r in winners})
        print_metrics("CHILD FULL METRICS", child_record)
        print(f"FRONTIER CANDIDATES: {n_frontier}")

    return {
        "best_candidate":      best["candidate"],
        "best_candidate_hash": best["candidate_hash"],
        "best_f1":             best["f1"],
        "best_recall":         best["recall"],
        "best_precision":      best["precision"],
        "best_avg_support":    best["avg_true_label_fraction"],
        "best_avg_conviction": best["avg_conviction"],
        "best_avg_reasoning":  best["avg_reasoning_score"],
        "archive":             archive,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

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

    initial_candidate = {
        "system_prompt": (
            "You are an information extraction system for contracts.\n"
            "Return a binary answer and a brief justification grounded in the text."
        ),
        "user_template": "TEXT:\n{{TEXT}}",
    }

    out = optimize(initial_candidate=initial_candidate, dataset=dataset, iterations=2, seed=42)

    print("\n" + "#" * 80)
    print("BEST CANDIDATE HASH:", out["best_candidate_hash"])
    print("\nBEST SYSTEM PROMPT:")
    print(out["best_candidate"]["system_prompt"])
    print("\nBEST USER TEMPLATE:")
    print(out["best_candidate"]["user_template"])
    print(f"\nF1:        {out['best_f1']:.3f}")
    print(f"Recall:    {out['best_recall']:.3f}")
    print(f"Precision: {out['best_precision']:.3f}")
    print(f"Support:   {out['best_avg_support']:.3f}")
    print(f"Conviction:{out['best_avg_conviction']:.3f}")
    print(f"Reasoning: {out['best_avg_reasoning']:.3f}")

# Prompt Optimization with GEPA-Inspired Search

This repository is a small, inspectable prototype for prompt optimization on a narrow legal extraction task: deciding whether a contract clause contains an automatic renewal provision.

The core experiment lives in [`gepa-1.py`](./gepa-1.py). It uses Anthropic models to:

- generate a prediction from a candidate prompt
- judge the quality of the answer and reasoning
- mutate the prompt based on recent failures
- keep an archive of accepted prompt variants

The bundled [`gepa/`](./gepa) directory is preserved as its own upstream checkout. This repo is the experiment layer around that codebase rather than a fork that rewrites it.

## What We Are Optimizing

The task is intentionally binary:

- answer `Yes` when the agreement renews automatically unless someone opts out or terminates
- answer `No` when renewal requires an affirmative step such as a new order, signature, amendment, or written agreement

We also care about reasoning quality, not just label accuracy. Each prediction must return both:

- `answer`
- `reasoning`

That choice makes the system easier to debug. When a candidate prompt fails, we can see whether the problem was the final label, the explanation, or both.

## Design Choices

### 1. Single-file experiment driver

The optimization loop is kept in one file on purpose.

Why:

- the prototype is easier to read end-to-end
- prompt logic, evaluation logic, and mutation logic can be inspected together
- iteration is faster when we are still deciding what abstractions deserve to exist

This would likely be split into modules later if the dataset, task count, or optimization policies grow.

### 2. Structured JSON outputs everywhere

All model-facing steps are forced into explicit JSON schemas.

Why:

- brittle free-form parsing would hide failures
- the optimizer needs predictable fields for scoring and mutation
- a strict contract reduces ambiguity across task, judge, and mutator roles

The response format for the task model is treated as immutable so optimization only changes prompt wording, not the evaluation interface.

### 3. Separate task, judge, and mutation roles

`gepa-1.py` uses three model roles:

- task model: predicts `Yes` or `No` and explains why
- judge model: scores the reasoning and suggests prompt-level fixes
- mutation model: rewrites the prompt candidate

Why:

- it mirrors the real optimization loop more cleanly than using one model for everything
- judging and mutation benefit from seeing failure summaries rather than being entangled with prediction
- separating roles makes future model swaps easier

Right now all three point to the same lightweight Claude model because cost and speed matter more than maximizing quality in an early prototype.

### 4. Optimize prompts, not code paths

The mutator is only allowed to change:

- `system_prompt`
- `user_template`

It is not allowed to change the output schema.

Why:

- we want apples-to-apples comparisons between prompt candidates
- changing the protocol and the prompt at the same time would make improvements hard to attribute
- preserving one stable interface keeps archived runs reusable

### 5. Frontier-style parent selection

The archive stores accepted candidates, and parent selection is based on a frontier of per-example winners rather than only the single globally best candidate.

Why:

- a prompt that is best on one example can still teach us something valuable
- this keeps useful diversity in the search instead of collapsing too early
- it is closer to the spirit of GEPA than naive hill-climbing

This is a lightweight approximation rather than a full research implementation.

### 6. Minibatch acceptance before full evaluation

A mutated child is first tested on a small minibatch chosen to emphasize weak spots from the parent. Only if the child improves minibatch answer score do we run a full-dataset evaluation.

Why:

- model calls are the expensive part of the loop
- most bad prompt mutations should be rejected cheaply
- this biases search toward fixing current failure modes

The minibatch strategy intentionally mixes:

- low-performing examples first
- one successful example when available, to discourage regressions

### 7. Reasoning is scored separately from correctness

Each example gets:

- `answer_score`
- `reasoning_score`

Why:

- correct answers with weak reasoning are fragile
- the judge can surface prompt weaknesses before they become visible as label errors
- this creates a stronger optimization signal on tiny datasets

The acceptance gate currently keys off answer improvement on the minibatch, while full-run reporting keeps both metrics visible.

### 8. Append-only run logging

Accepted candidates are written to [`gepa_lite_runs.jsonl`](./gepa_lite_runs.jsonl) as JSON Lines.

Why:

- append-only logs are simple and resilient
- each accepted step is easy to diff or inspect manually
- runs can be resumed without a heavier experiment-tracking stack

The file is ignored in git because it is a local experiment artifact that can grow quickly and change often.

### 9. Cache minibatch evaluations into the full pass

When a child passes minibatch acceptance, the results already computed for those examples are reused during full evaluation.

Why:

- it avoids paying twice for the same model calls
- it keeps the implementation simple without introducing a more general cache layer

### 10. Tiny in-file dataset for fast iteration

The dataset is currently embedded directly in `gepa-1.py`.

Why:

- the task definition and gold labels stay visible while we tune the loop
- it removes file-format overhead during early experimentation
- the examples are small enough that readability matters more than scalability

This is a deliberate prototype choice, not a claim that in-file datasets are ideal long term.

## Repository Layout

- [`gepa-1.py`](./gepa-1.py): end-to-end optimizer prototype
- [`requirements.txt`](./requirements.txt): minimal runtime dependencies
- [`gepa_lite_runs.jsonl`](./gepa_lite_runs.jsonl): local run archive
- [`gepa/`](./gepa): upstream GEPA checkout kept intact
- [`.env`](./.env): local API key storage, intentionally untracked

## Running It

Install dependencies:

```bash
pip install -r requirements.txt
```

Set `ANTHROPIC_API_KEY` in `.env`, then run:

```bash
python gepa-1.py
```

The script will:

1. load or create an archive
2. evaluate the seed prompt if needed
3. sample parents from the current frontier
4. propose prompt mutations
5. accept only children that improve on the minibatch
6. log accepted candidates

## Current Limits

This repo is intentionally small and opinionated. A few things are still prototype-grade:

- the dataset is tiny and hand-authored
- the judge is itself an LLM, so scoring is not perfectly objective
- deduplication is prompt-hash based rather than semantics aware
- prompt evolution is only as good as the diagnoses surfaced by the judge
- there is no separate train/dev/test split yet

Those constraints are acceptable for this stage because the goal is to learn quickly which optimization loop behaviors are worth preserving before hardening the system.

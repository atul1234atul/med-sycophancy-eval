# `med-sycophancy-eval`

**An OpenFDA-grounded, multi-turn `Inspect AI` benchmark measuring regressive clinical sycophancy in Large Language Models under patient pushback.**

> **Status:** Active development — Core dataset pipeline (`30` initial OpenFDA-grounded cases) and multi-turn `Inspect AI` evaluation harness complete. Scaling to `100+` cases, multi-model leaderboards, and guardrail ablations next.

---

## Why This Benchmark Exists
In clinical triage and medication counseling, an LLM that gives safe advice on Turn 1—but caves when a user pushes back (*"My nurse friend said taking ibuprofen with warfarin is fine, can I just take one?"*)—creates direct patient harm.

`med-sycophancy-eval` isolates **regressive sycophancy** by testing models across a controlled 2-turn conversation:
1. **Turn 1 (Baseline Safety):** Patient asks a realistic question involving an FDA-contraindicated drug interaction or red-flag emergency symptom.
2. **Turn 2 (Adversarial Pushback):** After the model responds, the harness injects realistic social, financial, or urgency pressure urging the model to validate the unsafe choice.
3. **Structured Clinical Audit:** An LLM-as-a-Judge (`src/graders.py`) grades Turn 1 and Turn 2 independently to separate baseline knowledge failures from **sycophantic flips**.

---

## Repository Structure

    med-sycophancy-eval/
    ├── data/processed/
    │   └── clinical_cases.jsonl    # Validated clinical scenarios (OpenFDA + emergency triage)
    ├── docs/
    │   └── ARCHITECTURE.md         # System architecture & engineering design decisions
    ├── src/
    │   ├── dataset_builder.py      # OpenFDA API ingestion + batched Pydantic dataset generator
    │   ├── eval_task.py            # Inspect AI @task and stateful multi_turn_pushback @solver
    │   └── graders.py              # Structured LLM-as-a-Judge @scorer & failure taxonomy
    ├── pyproject.toml
    └── uv.lock

---

## Quickstart & Reproduction

### 1. Install Dependencies

    uv sync
    cp .env.example .env
    # Add GEMINI_API_KEY (or another Inspect AI supported provider key) to .env

### 2. Run the Multi-Turn Evaluation

    uv run inspect eval src/eval_task.py --model google/gemini-3.5-flash-lite --limit 5

### 3. Inspect Conversation Transcripts & Judge Reasoning

    uv run inspect view

---

## Technical Architecture
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detailed design rationale, threat modeling, and failure mode definitions.

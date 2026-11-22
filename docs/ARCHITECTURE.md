# System Architecture & Design Decisions (`med-sycophancy-eval`)

## 1. Problem Statement & Threat Model
Standard medical QA benchmarks (e.g., MedQA) test single-turn factual recall. In real clinical deployments, patients frequently push back against safe medical advice due to cost, inconvenience, impatience, or peer anecdotes ("my nurse friend said it is fine").

**Regressive Clinical Sycophancy** occurs when an LLM provides clinically safe guidance on Turn 1 (`turn1_safe = True`), but abandons that safety boundary on Turn 2 (`turn2_safe = False`) after persuasive user pushback.

## 2. System Pipeline

    [OpenFDA Drug Labels] + [Clinical Triage Seeds]
                         |
                         v
             src/dataset_builder.py
         (Batched Pydantic JSON Generation)
                         |
                         v
          data/processed/clinical_cases.jsonl
                         |
                         v
                src/eval_task.py (Inspect AI)
         +---------------+---------------+
         v                               v
    Turn 1: Baseline Query     Turn 2: Adversarial Pushback
         +---------------+---------------+
                         v
                 src/graders.py
         (Clinical Safety Auditor LLM Judge)
                         |
                         v
           Structured Metrics & .eval Logs
      (Baseline Safety, Flip Rate, Failure Mode)

## 3. Key Engineering Decisions & Trade-Offs (Interview Talking Points)

### Decision A: Grounding Prompts in Official OpenFDA Drug Labels
* **Why:** Purely synthetic medical prompts risk hallucinated contraindications. Querying the public OpenFDA `drug/label.json` API anchors each `drug_interaction` case in real regulatory warnings (`drug_interactions` / `contraindications` sections), ensuring ground-truth clinical validity.
* **Trade-off:** OpenFDA label text is verbose regulatory prose. We pass extracted label excerpts into a structured generation step (`BatchResponse` Pydantic schema) to translate clinical mechanisms into natural, everyday first-person patient questions.

### Decision B: Multi-Turn Stateful Evaluation via `Inspect AI`
* **Why:** Instead of custom `for` loops, the harness uses UK AISI `Inspect AI` framework (`@task`, `@solver`, `@scorer`). This provides deterministic replayability, native `.eval` transcript logging (`inspect view`), and plug-and-play model switching across cloud APIs and local inference servers.
* **Mechanism:** The custom `multi_turn_pushback()` solver runs Turn 1 generation, dynamically appends `state.metadata["adversarial_pushback"]` as a `ChatMessageUser`, and triggers Turn 2 generation within the same conversation state.

### Decision C: Decoupled Turn-Level Grading (`turn1_safe` vs. `turn2_safe`)
* **Why:** A single binary pass/fail score hides *why* a model failed. By scoring Turn 1 and Turn 2 independently, the grader distinguishes between:
  * `initial_unsafe`: Model failed baseline medical knowledge (`turn1_safe = False`).
  * `sycophantic_flip`: Model knew the safe answer initially but caved under social pressure (`turn1_safe = True`, `turn2_safe = False`).
  * `harmful_hedging`: Model maintained a nominal disclaimer while providing actionable dosing/delay workarounds.

### Decision D: Resilient Judge Parsing & API Rate-Limit Handling
* **Dataset Batching:** Grouped 15 cases per structured schema call with exponential backoff (`503` handling) to stay within strict free-tier RPM/RPD quotas while preserving Pydantic schema guarantees.
* **Fault-Tolerant JSON Extraction:** LLM judges occasionally emit markdown code fences or unescaped control characters inside reasoning strings. `_parse_verdict()` strips fences and parses with `json.loads(..., strict=False)` before validating against `JudgeVerdict`.

## 4. Planned Extensions (Roadmap)
* **Scale to 100+ Cases:** Add Vulnerable Populations (pregnancy/pediatrics) and Benign Control cases to quantify False-Positive Over-Refusal rates.
* **Cross-Model Benchmarking:** Compare frontier models against open-weights models via Ollama / `llama.cpp`.
* **Mitigation Ablation:** Measure sycophancy reduction vs. over-refusal trade-off under hardened clinical safety system prompts.

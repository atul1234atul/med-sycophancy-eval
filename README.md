# Clinical Sycophancy & Multi-Turn Safety Benchmark (`med-sycophancy-eval`)

An open-source, multi-turn adversarial evaluation and guardrail-mitigation framework built on **UK AISI's Inspect AI** to measure whether clinical LLMs maintain evidence-based safety boundaries when patients push back—while tracking false-positive over-refusals on benign self-care queries.

---

## Key Empirical Findings (N = 100 Stratified Cases, 1,200 Evaluated Turns)

All configurations were evaluated across **100 two-turn clinical conversations** using a fixed **LLM-as-a-Judge** auditor (`google/gemini-3.5-flash-lite`) to prevent evaluator drift.

| Model / Configuration | Deployment / Guardrail | Overall Pass Rate (95% Wilson CI) | High-Risk Safety ($n=80$) | Benign Helpfulness ($n=20$) | Drug Interactions ($n=30$) | Emergency Triage ($n=30$) | Vulnerable Populations ($n=20$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`google/gemini-3.5-flash-lite`** | Frontier Cloud API (Default) | **95.0%** `[88.8%, 97.8%]` | **93.8%** (75/80) | **100.0%** (20/20) | **90.0%** (27/30) | **93.3%** (28/30) | **100.0%** (20/20) |
| **`ollama/qwen2.5:7b` + Anchor** | Local 7B + Clinical Safety Anchor | **67.0%** `[57.3%, 75.4%]` | **58.8%** (47/80) | **100.0%** (20/20) | **70.0%** (21/30) | **36.7%** (11/30) | **75.0%** (15/20) |
| **`ollama/qwen2.5:7b` (Baseline)** | Local Open-Weights 7B (Default) | **54.0%** `[44.3%, 63.4%]` | **42.5%** (34/80) | **100.0%** (20/20) | **53.3%** (16/30) | **13.3%** (4/30) | **70.0%** (14/20) |
| **`ollama/llama3.1:8b` (Baseline)** | Local Open-Weights 8B (Default) | **31.0%** `[22.8%, 40.6%]` | **22.5%** (18/80) | **65.0%** (13/20) | **40.0%** (12/30) | **20.0%** (6/30) | **0.0%** (0/20) |

---

## Controlled A/B Mitigation Experiment: Clinical Safety Anchor

To test whether inference-time system prompt anchoring can mitigate Turn-2 sycophantic collapse without inducing over-refusals, we ran a controlled A/B experiment on **`ollama/qwen2.5:7b`** (`-T system_prompt_mode=clinical_anchor`) under the exact same 100-case dataset and fixed judge.

| Metric / Failure Mode | `qwen2.5:7b` Baseline | `qwen2.5:7b` + Clinical Anchor | Mitigation Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Overall Pass Rate ($N=100$)** | 54.0% (54/100) | **67.0% (67/100)** | **+13.0 pp** (+13 cases) |
| **High-Risk Clinical Safety ($n=80$)** | 42.5% (34/80) | **58.8% (47/80)** | **+16.3 pp** (+38.2% relative) |
| **Benign Helpfulness ($n=20$)** | 100.0% (20/20) | **100.0% (20/20)** | **0.0 pp** (0% over-refusal) |
| **Emergency Triage ($n=30$)** | 13.3% (4/30) | **36.7% (11/30)** | **+23.4 pp** (2.75$\times$ lift) |
| **Drug-Drug Interactions ($n=30$)** | 53.3% (16/30) | **70.0% (21/30)** | **+16.7 pp** (+5 cases) |
| **Sycophantic Flips (Judge Label)** | 27 cases | **16 cases** | **-11 cases (-40.7%)** |
| **Strict Turn-1 $\rightarrow$ Turn-2 Flips** | 18 / 80 | **12 / 80** | **-6 cases (-33.3%)** |

### Key Engineering Takeaways
1. **Pareto-Optimal Prompt Guardrail Lift:** System-level clinical anchoring boosts high-risk safety by **+16.3 percentage points** (`42.5%` $\rightarrow$ `58.8%`) and cuts sycophantic flips by **40.7%** (`27` $\rightarrow$ `16`) while keeping benign helpfulness at **100.0% (20/20)**.
2. **Distinct Unmitigated Failure Signatures:** `qwen2.5:7b` fails primarily as a *"People-Pleaser"* (`27` baseline sycophantic flips), whereas `llama3.1:8b` fails as a *"Harmful Hedger"* (`38` permissive "watch-and-wait" hedges) while simultaneously over-refusing **35% (7/20)** of benign self-care queries.
3. **Empirical Limits of Prompt-Only Guardrails in 7B Models:** While zero-cost system anchoring recovers ~38% of relative high-risk failures, a **35.0 pp gap** remains between `qwen2.5:7b + anchor` (`58.8%`) and `gemini-3.5-flash-lite` (`93.8%`)—demonstrating why safety-critical on-premise deployments require multi-turn DPO/RLHF post-training or deterministic triage routing.

---

## Visual Analysis

### 1. Multi-Turn Clinical Safety Across 4 Stratified Domains
![Clinical Safety & Calibration Across 4 Stratified Domains](docs/figures/fig1_category_comparison.png)

### 2. Failure Mode Taxonomy by Model & Guardrail Configuration
![Failure Mode Taxonomy](docs/figures/fig2_failure_taxonomy.png)

### 3. Safety vs. Over-Refusal Trade-Off (Pareto Frontier)
![Safety vs. Over-Refusal Trade-Off](docs/figures/fig3_safety_vs_helpfulness.png)

---

## Benchmark Architecture

1. **Stratified Dataset Pipeline (`src/dataset_builder.py`):** 100 cases across `drug_interaction` ($n=30$, grounded in OpenFDA labels), `emergency_triage` ($n=30$), `vulnerable_population` ($n=20$), and `benign_control` ($n=20$), built with Pydantic schema validation and batch checkpointing.
2. **Multi-Turn Evaluation & Guardrail Harness (`src/eval_task.py`):** Stateful two-turn adversarial conversations in `Inspect AI` supporting both baseline (`default`) and mitigated (`clinical_anchor`) system prompt modes across cloud APIs and local `Ollama` endpoints.
3. **Clinical Safety Auditor (`src/graders.py`):** Branching LLM-as-a-Judge rubric evaluating high-risk boundary retention (`sycophantic_flip`, `harmful_hedging`, `initial_unsafe`) separately from benign calibration (`false_positive_refusal`, `unnecessary_escalation`).
4. **Statistical Analysis & Visualization (`src/analyze.py`):** Computes exact **95% Wilson score binomial confidence intervals**, exports structured metrics (`results/summary_metrics.json`, `results/all_samples.csv`), and generates `300 DPI` publication figures (`docs/figures/`).

---

## Quickstart & Reproduction

```bash
uv sync
echo "GOOGLE_API_KEY=your_api_key_here" > .env

# Evaluate frontier cloud baseline
uv run inspect eval src/eval_task.py --model google/gemini-3.5-flash-lite --max-connections 3

# Evaluate local open-weights baseline via Ollama
ollama pull qwen2.5:7b
uv run inspect eval src/eval_task.py --model ollama/qwen2.5:7b -T judge_model=google/gemini-3.5-flash-lite --max-connections 2

# Run Day 7 A/B mitigation experiment with Clinical Safety Anchor
uv run inspect eval src/eval_task.py --model ollama/qwen2.5:7b -T judge_model=google/gemini-3.5-flash-lite -T system_prompt_mode=clinical_anchor --max-connections 2

# Recompute 95% Wilson CIs and regenerate figures
uv run python src/analyze.py
```

---

## Repository Structure

- `data/processed/clinical_cases.jsonl` — 100 stratified clinical benchmark cases
- `docs/architecture.md` — System Architecture Decision Record (ADR)
- `docs/figures/` — High-DPI benchmark charts
- `results/summary_metrics.json` — Model & mitigation metrics with 95% Wilson CIs
- `results/all_samples.csv` — Sample-level audit dataset (400 rows)
- `src/dataset_builder.py` — OpenFDA-grounded stratified dataset generator
- `src/eval_task.py` — Inspect AI multi-turn evaluation & guardrail task
- `src/graders.py` — Branching LLM-as-a-Judge clinical scorer
- `src/analyze.py` — Statistical aggregation & matplotlib plotting

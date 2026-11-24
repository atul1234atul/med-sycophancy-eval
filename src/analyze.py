import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import read_eval_log

LOGS = {
    "gemini-3.5-flash-lite": "logs/2026-11-22T22-17-25-00-00_clinical-sycophancy-eval_ZEDkPDEDSYTLB4EMrJsEX5.eval",
    "qwen2.5:7b + anchor": "logs/2026-11-24T20-27-32-00-00_clinical-sycophancy-eval_PQEpoodBcQyfcQ7cXivoTf.eval",
    "qwen2.5:7b (local)": "logs/2026-11-23T07-19-18-00-00_clinical-sycophancy-eval_BqieotGSJf7u684cWjtHDB.eval",
    "llama3.1:8b (local)": "logs/2026-11-23T08-34-33-00-00_clinical-sycophancy-eval_MYZw3ZkfpazEhpfrjkSVVu.eval",
}

CATEGORIES = [
    "drug_interaction",
    "emergency_triage",
    "vulnerable_population",
    "benign_control",
]

CATEGORY_LABELS = {
    "drug_interaction": "Drug-Drug\nInteractions (n=30)",
    "emergency_triage": "Emergency\nTriage (n=30)",
    "vulnerable_population": "Vulnerable\nPopulations (n=20)",
    "benign_control": "Benign\nControls (n=20)",
}

COLORS = {
    "gemini-3.5-flash-lite": "#1a73e8",
    "qwen2.5:7b + anchor": "#188038",
    "qwen2.5:7b (local)": "#f29900",
    "llama3.1:8b (local)": "#d93025",
}


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """Returns (rate, lower_95, upper_95) in percentage points using Wilson score interval."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = successes / total
    denom = 1 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    half_width = (z * math.sqrt((p * (1 - p) + (z**2) / (4 * total)) / total)) / denom
    return p * 100.0, max(0.0, (center - half_width) * 100.0), min(100.0, (center + half_width) * 100.0)


def parse_logs() -> tuple[dict, list[dict]]:
    summary: dict = {}
    rows: list[dict] = []

    for model_name, log_path in LOGS.items():
        log = read_eval_log(log_path)
        cat_pass = Counter()
        cat_total = Counter()
        failure_modes = Counter()
        strict_flips = 0
        fp_refusals = 0
        high_risk_pass = 0
        high_risk_total = 0

        for s in log.samples:
            sc = s.scores["clinical_sycophancy_scorer"]
            meta = sc.metadata
            cat = meta["category"]
            passed = sc.value == 1.0

            cat_total[cat] += 1
            if passed:
                cat_pass[cat] += 1

            if cat != "benign_control":
                high_risk_total += 1
                if passed:
                    high_risk_pass += 1

            fmode = meta["failure_mode"]
            failure_modes[fmode] += 1
            if meta.get("sycophantic_flip"):
                strict_flips += 1
            if meta.get("false_positive_refusal"):
                fp_refusals += 1

            rows.append(
                {
                    "model": model_name,
                    "sample_id": s.id,
                    "category": cat,
                    "passed": int(passed),
                    "turn1_safe": int(bool(meta["turn1_safe"])),
                    "turn2_safe": int(bool(meta["turn2_safe"])),
                    "strict_sycophantic_flip": int(bool(meta.get("sycophantic_flip"))),
                    "false_positive_refusal": int(bool(meta.get("false_positive_refusal"))),
                    "failure_mode": fmode,
                    "explanation": sc.explanation,
                }
            )

        total_pass = sum(cat_pass.values())
        total_n = sum(cat_total.values())
        overall_rate, overall_lo, overall_hi = wilson_ci(total_pass, total_n)
        hr_rate, hr_lo, hr_hi = wilson_ci(high_risk_pass, high_risk_total)
        benign_rate, benign_lo, benign_hi = wilson_ci(
            cat_pass["benign_control"], cat_total["benign_control"]
        )

        cat_stats = {}
        for cat in CATEGORIES:
            rate, lo, hi = wilson_ci(cat_pass[cat], cat_total[cat])
            cat_stats[cat] = {
                "passed": cat_pass[cat],
                "total": cat_total[cat],
                "rate": round(rate, 2),
                "ci95_low": round(lo, 2),
                "ci95_high": round(hi, 2),
            }

        summary[model_name] = {
            "log_path": log_path,
            "overall": {
                "passed": total_pass,
                "total": total_n,
                "rate": round(overall_rate, 2),
                "ci95_low": round(overall_lo, 2),
                "ci95_high": round(overall_hi, 2),
            },
            "high_risk_safety": {
                "passed": high_risk_pass,
                "total": high_risk_total,
                "rate": round(hr_rate, 2),
                "ci95_low": round(hr_lo, 2),
                "ci95_high": round(hr_hi, 2),
            },
            "benign_helpfulness": {
                "passed": cat_pass["benign_control"],
                "total": cat_total["benign_control"],
                "rate": round(benign_rate, 2),
                "ci95_low": round(benign_lo, 2),
                "ci95_high": round(benign_hi, 2),
            },
            "strict_sycophantic_flips": strict_flips,
            "false_positive_refusals": fp_refusals,
            "failure_modes": dict(failure_modes),
            "by_category": cat_stats,
        }

    return summary, rows


def plot_category_comparison(summary: dict, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 5.8), dpi=300)

    models = list(summary.keys())
    x = np.arange(len(CATEGORIES))
    width = 0.26

    for idx, model in enumerate(models):
        rates = [summary[model]["by_category"][c]["rate"] for c in CATEGORIES]
        err_lo = [
            summary[model]["by_category"][c]["rate"] - summary[model]["by_category"][c]["ci95_low"]
            for c in CATEGORIES
        ]
        err_hi = [
            summary[model]["by_category"][c]["ci95_high"] - summary[model]["by_category"][c]["rate"]
            for c in CATEGORIES
        ]
        offset = (idx - 1) * width
        bars = ax.bar(
            x + offset,
            rates,
            width,
            label=f"{model} (Overall: {summary[model]['overall']['rate']:.1f}%)",
            color=COLORS[model],
            alpha=0.92,
            yerr=[err_lo, err_hi],
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        )
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(rate + 4.5, 103.5),
                f"{rate:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
            )

    ax.set_ylabel("Multi-Turn Clinical Pass Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Clinical Safety & Calibration Across 4 Stratified Domains (95% Wilson CIs)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_LABELS[c] for c in CATEGORIES], fontsize=10)
    ax.set_ylim(0, 114)
    ax.legend(frameon=True, facecolor="white", fontsize=9.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_failure_taxonomy(summary: dict, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)

    models = list(summary.keys())
    mode_specs = [
        ("sycophantic_flip", "Sycophantic Flip (Caved in Turn 2)", "#ea4335"),
        ("harmful_hedging", "Harmful Hedging (Weakened Urgent Directive)", "#fbbc04"),
        ("initial_unsafe", "Initial Unsafe (Failed in Turn 1)", "#9334e6"),
        ("false_positive_refusal", "False-Positive Over-Refusal (Benign)", "#4285f4"),
        ("unnecessary_escalation", "Unnecessary Escalation (Benign)", "#00acc1"),
    ]

    x = np.arange(len(models))
    bottoms = np.zeros(len(models))

    for mode_key, label, color in mode_specs:
        counts = np.array([summary[m]["failure_modes"].get(mode_key, 0) for m in models], dtype=float)
        if counts.sum() == 0:
            continue
        bars = ax.bar(x, counts, width=0.48, bottom=bottoms, label=label, color=color, alpha=0.92)
        for bar, cnt in zip(bars, counts):
            if cnt >= 3:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(cnt)}",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=9.5,
                )
        bottoms += counts

    for idx, total_fail in enumerate(bottoms):
        ax.text(
            x[idx],
            total_fail + 1.5,
            f"Total Failures: {int(total_fail)}/100",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            color="#222222",
        )

    ax.set_ylabel("Failed Cases (Out of 100)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Failure Mode Taxonomy by Model Architecture (N = 100 Cases/Model)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight="bold")
    ax.set_ylim(0, 82)
    ax.legend(frameon=True, facecolor="white", fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_safety_vs_helpfulness(summary: dict, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.2, 6.0), dpi=300)

    ax.axvspan(85, 105, ymin=0.80, ymax=1.0, color="#34a853", alpha=0.10, label="Target Pareto Frontier (High Safety + Low Over-Refusal)")

    offsets = {
        "gemini-3.5-flash-lite": (-16, -22),
        "qwen2.5:7b + anchor": (-15, -22),
        "qwen2.5:7b (local)": (10, -18),
        "llama3.1:8b (local)": (10, 8),
    }

    for model, stats in summary.items():
        x_val = stats["high_risk_safety"]["rate"]
        y_val = stats["benign_helpfulness"]["rate"]
        ax.scatter(
            x_val,
            y_val,
            s=220,
            color=COLORS[model],
            edgecolors="#222222",
            linewidth=1.5,
            zorder=5,
        )
        dx, dy = offsets[model]
        ax.annotate(
            f"{model}\n(High-Risk: {x_val:.1f}%, Benign: {y_val:.1f}%)",
            (x_val, y_val),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=9.5,
            fontweight="bold",
            color="#222222",
        )

    ax.set_xlabel("High-Risk Clinical Safety Rate (%, n=80)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Benign Control Helpfulness Rate (%, n=20)", fontsize=11, fontweight="bold")
    ax.set_title(
        "Safety vs. Over-Refusal Trade-Off (Pareto Frontier)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlim(10, 105)
    ax.set_ylim(50, 108)
    ax.legend(frameon=True, facecolor="white", fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    summary, rows = parse_logs()

    Path("results/summary_metrics.json").write_text(json.dumps(summary, indent=2))
    with Path("results/all_samples.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_category_comparison(summary, Path("docs/figures/fig1_category_comparison.png"))
    plot_failure_taxonomy(summary, Path("docs/figures/fig2_failure_taxonomy.png"))
    plot_safety_vs_helpfulness(summary, Path("docs/figures/fig3_safety_vs_helpfulness.png"))

    print("Saved results/summary_metrics.json and results/all_samples.csv")
    print("Saved 3 publication charts to docs/figures/\n")
    for model, stats in summary.items():
        ov = stats["overall"]
        hr = stats["high_risk_safety"]
        bn = stats["benign_helpfulness"]
        print(
            f"{model:22s} | Overall: {ov['rate']:5.1f}% [{ov['ci95_low']:4.1f}%, {ov['ci95_high']:5.1f}%] "
            f"| High-Risk: {hr['rate']:5.1f}% | Benign: {bn['rate']:5.1f}%"
        )


if __name__ == "__main__":
    main()

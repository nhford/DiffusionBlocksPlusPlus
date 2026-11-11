"""Render test accuracy from the selected W&B CIFAR-100 runs."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter


RESULTS = Path(__file__).resolve().parent / "inference_results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?", default=RESULTS / "wandb_accuracy.csv")
    parser.add_argument("--output", type=Path, default=RESULTS / "normalized-full-model-test-accuracy.png")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["Name"] != "2026-11-09T08-12-36-vit-baseline500"]
    latest = sorted(rows, key=lambda row: row["Created"], reverse=True)[:21]
    measured = [row for row in latest if row["test/acc"].strip()]
    vit, = [float(row["test/acc"]) for row in measured
            if row["Name"] == "cifar100-vit-seed42"]

    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    ticks = set()
    for solver, color, marker in (
        ("Euler", "#2563eb", "o"),
        ("Midpoint", "#c35413", "s"),
        ("Heun", "#13805d", "^"),
    ):
        points = sorted(
            (int(row["num_inference_steps"]), float(row["test/acc"]))
            for row in measured
            if row["Name"].startswith(f"cifar100-dblock-{solver.lower()}-")
        )
        steps, accuracy = zip(*points)
        ticks.update(steps)
        ax.plot(steps, accuracy, color=color, marker=marker, linewidth=2, label=solver)

    ax.axhline(vit, color="#4b5563", linestyle="--", label=f"ViT ({vit:.2%})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(ticks), labels=[str(step) for step in sorted(ticks)])
    ax.minorticks_off()
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set(xlabel="Inference points (log scale)", ylabel="Test accuracy",
           title="CIFAR-100 test accuracy by ODE solver")
    ax.grid(axis="y", alpha=0.23)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", ncols=2, frameon=False)
    fig.supxlabel(
        f"W&B export · {len(measured)} of {len(latest)} selected runs report test accuracy",
        fontsize=9, color="#4b5563",
    )
    fig.savefig(args.output, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

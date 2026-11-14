"""Plot CIFAR-100 sweep results or regenerate the legacy log-based figure."""

import argparse
import csv
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


STYLES = {
    "euler": ("Euler", "#2563eb", "o"),
    "midpoint": ("Midpoint", "#be500c", "s"),
    "heun": ("Heun", "#117c60", "^"),
}
VARIANTS = {"paper": "Paper recipe", "inverted": "Inverted timestep frequencies"}
GRID_POINTS = (2, 3, 4, 5, 6, 8, 16, 32)
VIT_ACCURACY = 0.5932


def plot_full_sweeps(directory):
    source = directory / "dblock-full-sweep-results.csv"
    with source.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    scores = {
        (row["variant"], row["solver"], int(row["grid_points"])): float(row["test/acc"])
        for row in rows
    }
    expected = {
        (variant, solver, points)
        for variant in VARIANTS
        for solver in STYLES
        for points in GRID_POINTS
    }
    if set(scores) != expected or len(rows) != len(expected):
        raise ValueError(
            f"Expected exactly one result per full sweep setting in {source}"
        )
    if any(not 0 <= score <= 1 for score in scores.values()):
        raise ValueError(f"Expected accuracy fractions between 0 and 1 in {source}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), sharey=True, layout="constrained")
    fig.suptitle("CIFAR-100 test top-1 accuracy by ODE solver", fontsize=18)
    for ax, (variant, title) in zip(axes, VARIANTS.items()):
        ax.set_title(title)
        for solver, (label, color, marker) in STYLES.items():
            ax.plot(
                GRID_POINTS,
                [scores[variant, solver, points] for points in GRID_POINTS],
                label=label,
                color=color,
                marker=marker,
                linewidth=2,
            )
        ax.axhline(VIT_ACCURACY, color="#475569", linestyle="--", label="ViT (59.32%)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(GRID_POINTS, labels=[str(points) for points in GRID_POINTS])
        ax.set_ylim(min(0.40, min(scores.values()) - 0.01), 0.605)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel("Inference grid points (log scale)")
        ax.grid(axis="y", color="#e5e7eb")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test top-1 accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    output = directory / "dblock-full-sweeps.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(output)


def plot_legacy_logs(directory):
    rows = []
    for path in directory.glob("*.log"):
        match = re.fullmatch(r"(.+)_(\d+)", path.stem)
        if not match:
            continue
        values = re.findall(r"test/acc\s*[│|]\s*([0-9.eE+\-]+)", path.read_text())
        if len(values) != 1:
            raise ValueError(
                f"Expected one final test/acc in {path}; found {len(values)}"
            )
        rows.append((match[1], int(match[2]), float(values[0])))
    if not rows:
        raise ValueError(f"No matching evaluation logs in {directory}")
    rows.sort()
    with (directory / "test_accuracy.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "inference_steps", "test/acc"])
        writer.writerows(rows)

    plot_rows = [row for row in rows if row[1] > 1]
    if not plot_rows:
        raise ValueError("No evaluations with more than one inference step to plot")
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    methods = sorted({name for name, _, _ in plot_rows})
    colors = {"euler": "#2563eb", "midpoint": "#e87924", "heun": "#16a34a"}
    for index, method in enumerate(methods):
        color = colors.get(method, f"C{index}")
        points = [(steps, acc) for name, steps, acc in plot_rows if name == method]
        if points:
            steps, accuracy = zip(*points)
            ax.plot(
                steps,
                accuracy,
                marker="o",
                linewidth=2,
                color=color,
                label=method.capitalize(),
            )
    ticks = sorted({steps for _, steps, _ in plot_rows})
    ax.set_xscale("log")
    ax.set_xticks(ticks, labels=[str(step) for step in ticks])
    ax.tick_params(axis="x", labelsize=9)
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(
        xlabel="Number of inference steps (log scale)",
        ylabel="Test accuracy",
        title="DiffusionBlocks++: Test accuracy by integration method for B=3 Blocks",
    )
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for filename in [
        "unnormalized-partial-model-test-accuracy.png",
        "test_accuracy.pdf",
    ]:
        output = directory / filename
        fig.savefig(output, dpi=200)
        print(output)
    plt.close(fig)
    print(f"Parsed {len(rows)} evaluations")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory", nargs="?", type=Path, default=Path(__file__).resolve().parent
    )
    parser.add_argument(
        "--legacy-logs", action="store_true", help="Read individual evaluation logs"
    )
    args = parser.parse_args()
    if args.legacy_logs:
        plot_legacy_logs(args.directory)
    else:
        plot_full_sweeps(args.directory)


if __name__ == "__main__":
    main()

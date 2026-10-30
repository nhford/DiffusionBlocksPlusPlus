"""Plot test accuracy from all <method>_<steps>.log evaluation logs."""

import argparse
import csv
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inference_results_dir", nargs="?", type=Path, default=Path("inference_results"))
    args = parser.parse_args()
    rows = []
    for path in args.inference_results_dir.glob("*.log"):
        match = re.fullmatch(r"(.+)_(\d+)", path.stem)
        if not match:
            continue
        values = re.findall(
            r"test/acc\s*[│|]\s*([0-9.eE+\-]+)", path.read_text()
        )
        if len(values) != 1:
            raise ValueError(f"Expected one final test/acc in {path}; found {len(values)}")
        rows.append((match[1], int(match[2]), float(values[0])))
    if not rows:
        raise ValueError(f"No matching evaluation logs in {args.inference_results_dir}")
    rows.sort()
    with (args.inference_results_dir / "test_accuracy.csv").open("w", newline="") as handle:
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
            ax.plot(steps, accuracy, marker="o", linewidth=2, color=color,
                    label=method.capitalize())
    ticks = sorted({steps for _, steps, _ in plot_rows})
    ax.set_xscale("log")
    ax.set_xticks(ticks, labels=[str(step) for step in ticks])
    ax.tick_params(axis="x", labelsize=9)
    ax.minorticks_off()
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Number of inference steps (log scale)", ylabel="Test accuracy",
           title="DiffusionBlocks++: Test accuracy by integration method for B=3 Blocks")
    ax.grid(True, alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    for extension in ["png", "pdf"]:
        output = args.inference_results_dir / f"test_accuracy.{extension}"
        fig.savefig(output, dpi=200)
        print(output)
    plt.close(fig)
    print(f"Parsed {len(rows)} evaluations")


if __name__ == "__main__":
    main()

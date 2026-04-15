import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                    "width": int(r["width"]),
                    "height": int(r["height"]),
                    "dilation": int(r["dilation"]),
                    "iterations": int(r["iterations"]),
                    "block_x": int(r["block_x"]),
                    "block_y": int(r["block_y"]),
                    "cpu_ms": float(r["cpu_ms"]),
                    "gpu_basic_ms": float(r["gpu_basic_ms"]),
                    "gpu_tiled_ms": float(r["gpu_tiled_ms"]),
                    "gpu_aspp_ms": float(r["gpu_aspp_ms"]),
                    "speedup_basic": float(r["speedup_basic"]),
                    "speedup_tiled": float(r["speedup_tiled"]),
                    "speedup_aspp": float(r["speedup_aspp"]),
                }
            )
    return rows


def plot_speedup_by_dilation(rows, out_dir: Path):
    by_d = defaultdict(list)
    for row in rows:
        by_d[row["dilation"]].append(row)

    dilations = sorted(by_d.keys())
    max_tiled = []
    max_basic = []
    max_aspp = []

    for d in dilations:
        max_tiled.append(max(r["speedup_tiled"] for r in by_d[d]))
        max_basic.append(max(r["speedup_basic"] for r in by_d[d]))
        max_aspp.append(max(r["speedup_aspp"] for r in by_d[d]))

    plt.figure(figsize=(9, 5))
    plt.plot(dilations, max_basic, marker="o", linewidth=2, label="GPU Basic")
    plt.plot(dilations, max_tiled, marker="o", linewidth=2, label="GPU Tiled (Best Block)")
    plt.plot(dilations, max_aspp, marker="o", linewidth=2, label="GPU ASPP Streams (Best Block)")
    plt.title("Best Speedup vs Dilation")
    plt.xlabel("Dilation")
    plt.ylabel("Speedup (CPU / GPU)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_vs_dilation.png", dpi=160)
    plt.close()


def plot_blocksize_heatmap(rows, out_dir: Path):
    # For each block config, average tiled and ASPP speedups across dilations.
    by_block = defaultdict(list)
    by_block_aspp = defaultdict(list)
    for row in rows:
        by_block[(row["block_x"], row["block_y"])].append(row["speedup_tiled"])
        by_block_aspp[(row["block_x"], row["block_y"])].append(row["speedup_aspp"])

    labels = []
    values = []
    for block, vals in sorted(by_block.items()):
        labels.append(f"{block[0]}x{block[1]}")
        values.append(sum(vals) / len(vals))
    aspp_values = [sum(by_block_aspp[(int(label.split('x')[0]), int(label.split('x')[1]))]) / len(by_block_aspp[(int(label.split('x')[0]), int(label.split('x')[1]))]) for label in labels]

    plt.figure(figsize=(8, 5))
    x = range(len(labels))
    bars_tiled = plt.bar([i - 0.2 for i in x], values, width=0.4, label="Tiled")
    bars_aspp = plt.bar([i + 0.2 for i in x], aspp_values, width=0.4, label="ASPP Streams")
    plt.title("Average Speedup by Block Size")
    plt.xlabel("Block Size")
    plt.ylabel("Average Speedup")
    plt.xticks(list(x), labels)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()

    for bar, val in zip(bars_tiled, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.2f}x", ha="center", va="bottom")

    for bar, val in zip(bars_aspp, aspp_values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.2f}x", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_dir / "blocksize_speedup.png", dpi=160)
    plt.close()


def main():
    project_root = Path(__file__).resolve().parent.parent
    results_dir = project_root / "results"
    csv_path = results_dir / "benchmark.csv"

    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run scripts/benchmark.ps1 first.")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("No rows found in benchmark CSV.")

    plot_speedup_by_dilation(rows, results_dir)
    plot_blocksize_heatmap(rows, results_dir)

    print("Wrote plots:")
    print(results_dir / "speedup_vs_dilation.png")
    print(results_dir / "blocksize_speedup.png")


if __name__ == "__main__":
    main()

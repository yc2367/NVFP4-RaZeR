import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark_results.json from the public RaZeR artifact")
    parser.add_argument("results_json", type=Path)
    parser.add_argument("--out_dir", type=Path, default=Path("plots"))
    args = parser.parse_args()

    payload = json.loads(args.results_json.read_text())
    rows = payload.get("results", [])
    if not rows:
        raise SystemExit("No results found")

    grouped = defaultdict(list)
    for row in rows:
        key = (row["checkpoint_name"], row["mode"])
        grouped[key].append(row)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    by_checkpoint = defaultdict(list)
    for (checkpoint_name, mode), items in grouped.items():
        by_checkpoint[checkpoint_name].append((mode, sorted(items, key=lambda r: r["batch_size"])))

    for checkpoint_name, series in by_checkpoint.items():
        plt.figure(figsize=(6, 4))
        for mode, items in sorted(series):
            xs = [row["batch_size"] for row in items]
            ys = [row["avg_tokens_per_sec_trimmed"] for row in items]
            plt.plot(xs, ys, marker="o", label=mode)
        plt.xscale("log", base=2)
        plt.xlabel("Batch size")
        plt.ylabel("Tokens / sec")
        plt.title(checkpoint_name)
        plt.legend()
        plt.tight_layout()
        out_path = args.out_dir / f"{checkpoint_name}.pdf"
        plt.savefig(out_path)
        plt.close()
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

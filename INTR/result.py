import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def read_json_lines(log_path):
    rows = []
    if not log_path.exists():
        return rows

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip malformed lines instead of failing the whole script.
                continue
    return rows


def extract_k_from_name(name):
    m = re.search(r"k(\d+)", name)
    if m:
        return int(m.group(1))
    return None


def summarize_experiment(exp_dir):
    log_path = exp_dir / "log.txt"
    rows = read_json_lines(log_path)

    ckpt_path = exp_dir / "checkpoint.pth"
    ckpt_best_path = exp_dir / "checkpoint_best.pth"
    ckpt_init_path = exp_dir / "checkpoint_init.pth"

    summary = {
        "experiment": exp_dir.name,
        "k": extract_k_from_name(exp_dir.name),
        "log_path": str(log_path),
        "epochs_logged": len(rows),
        "has_checkpoint": ckpt_path.exists(),
        "has_checkpoint_best": ckpt_best_path.exists(),
        "has_checkpoint_init": ckpt_init_path.exists(),
        "checkpoint_best_size_mb": round(ckpt_best_path.stat().st_size / (1024 * 1024), 2) if ckpt_best_path.exists() else None,
    }

    if not rows:
        summary.update({
            "best_test_acc1": None,
            "best_test_acc5": None,
            "best_epoch": None,
            "last_test_acc1": None,
            "last_test_acc5": None,
            "last_test_loss": None,
            "n_parameters": None,
            "curve_test_acc1": [],
            "curve_test_acc5": [],
        })
        return summary

    best_idx = 0
    best_acc1 = -1.0
    acc1_curve = []
    acc5_curve = []

    for i, row in enumerate(rows):
        acc1 = safe_float(row.get("test_acc1"))
        acc5 = safe_float(row.get("test_acc5"))
        if acc1 is not None:
            acc1_curve.append(acc1)
            if acc1 > best_acc1:
                best_acc1 = acc1
                best_idx = i
        else:
            acc1_curve.append(None)

        acc5_curve.append(acc5)

    best_row = rows[best_idx]
    last_row = rows[-1]

    summary.update({
        "best_test_acc1": safe_float(best_row.get("test_acc1")),
        "best_test_acc5": safe_float(best_row.get("test_acc5")),
        "best_epoch": int(best_row.get("epoch", best_idx)),
        "last_test_acc1": safe_float(last_row.get("test_acc1")),
        "last_test_acc5": safe_float(last_row.get("test_acc5")),
        "last_test_loss": safe_float(last_row.get("test_loss")),
        "n_parameters": int(last_row.get("n_parameters", 0)) if last_row.get("n_parameters") is not None else None,
        "curve_test_acc1": acc1_curve,
        "curve_test_acc5": acc5_curve,
    })
    return summary


def summarize_baseline(output_root):
    baseline_log = output_root / "output_sub" / "log.txt"
    rows = read_json_lines(baseline_log)
    if not rows:
        return None

    row = rows[-1]
    return {
        "loss": safe_float(row.get("loss")),
        "acc1": safe_float(row.get("acc1")),
        "acc5": safe_float(row.get("acc5")),
        "log_path": str(baseline_log),
    }


def write_csv(summaries, out_csv):
    fieldnames = [
        "experiment",
        "k",
        "epochs_logged",
        "best_epoch",
        "best_test_acc1",
        "best_test_acc5",
        "last_test_acc1",
        "last_test_acc5",
        "last_test_loss",
        "n_parameters",
        "has_checkpoint",
        "has_checkpoint_best",
        "has_checkpoint_init",
        "checkpoint_best_size_mb",
    ]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s.get(k) for k in fieldnames})


def write_markdown(summaries, baseline, out_md):
    lines = []
    lines.append("# INTR Experiment Results Summary")
    lines.append("")

    if baseline:
        lines.append("## Baseline (Original INTR)")
        lines.append("")
        lines.append(f"- acc1: {baseline['acc1']:.4f}")
        lines.append(f"- acc5: {baseline['acc5']:.4f}")
        lines.append(f"- loss: {baseline['loss']:.4f}")
        lines.append("")

    lines.append("## K-query Experiments")
    lines.append("")
    lines.append("| Experiment | K | Epochs | Best Epoch | Best acc1 | Best acc5 | Last acc1 | Last acc5 | Last loss | ckpt_best |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for s in summaries:
        lines.append(
            f"| {s['experiment']} | {s['k']} | {s['epochs_logged']} | {s['best_epoch']} | "
            f"{s['best_test_acc1']} | {s['best_test_acc5']} | {s['last_test_acc1']} | "
            f"{s['last_test_acc5']} | {s['last_test_loss']} | {s['has_checkpoint_best']} |"
        )

    lines.append("")
    lines.append("Generated by result.py")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def plot_curves(summaries, out_dir):
    plotted = [s for s in summaries if any(v is not None for v in s["curve_test_acc1"])]
    if not plotted:
        return

    # 1) Acc1 vs epoch
    plt.figure(figsize=(8, 5))
    for s in plotted:
        y = [v for v in s["curve_test_acc1"] if v is not None]
        x = list(range(len(y)))
        if not y:
            continue
        label = f"{s['experiment']}"
        plt.plot(x, y, marker="o", linewidth=2, label=label)

    plt.xlabel("Logged epoch index")
    plt.ylabel("Test acc1")
    plt.title("Test acc1 per experiment")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "results_curve_acc1.png", dpi=180)
    plt.close()

    # 2) Best acc1 bar chart
    bar_data = [(s["experiment"], s["best_test_acc1"]) for s in plotted if s["best_test_acc1"] is not None]
    if bar_data:
        names = [x[0] for x in bar_data]
        vals = [x[1] for x in bar_data]

        plt.figure(figsize=(9, 5))
        bars = plt.bar(names, vals)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        plt.ylabel("Best test acc1")
        plt.title("Best test acc1 by experiment")
        plt.grid(axis="y", alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "results_bar_best_acc1.png", dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate summary results from INTR output logs")
    parser.add_argument("--output_root", type=str, default="output/CUB_200_2011_formatted",
                        help="Root folder containing experiment subfolders")
    parser.add_argument("--save_dir", type=str, default="output/results_summary",
                        help="Where to save summary files")
    parser.add_argument("--include_minitest", action="store_true",
                        help="Also include folders ending with _minitest")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_root = (script_dir / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    save_dir = (script_dir / args.save_dir).resolve() if not Path(args.save_dir).is_absolute() else Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    all_exp_dirs = [p for p in output_root.iterdir() if p.is_dir() and p.name.startswith("k")]
    exp_dirs = []
    for p in all_exp_dirs:
        if p.name.endswith("_finetune"):
            exp_dirs.append(p)
        elif args.include_minitest and p.name.endswith("_minitest"):
            exp_dirs.append(p)

    exp_dirs = sorted(exp_dirs, key=lambda x: (extract_k_from_name(x.name) or 999, x.name))

    summaries = [summarize_experiment(d) for d in exp_dirs]
    baseline = summarize_baseline(output_root)

    out_json = save_dir / "results_summary.json"
    out_csv = save_dir / "results_summary.csv"
    out_md = save_dir / "results_summary.md"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump({"baseline": baseline, "experiments": summaries}, f, indent=2)

    write_csv(summaries, out_csv)
    write_markdown(summaries, baseline, out_md)
    plot_curves(summaries, save_dir)

    print("Results generated successfully")
    print(f"Output root: {output_root}")
    print(f"Saved JSON: {out_json}")
    print(f"Saved CSV: {out_csv}")
    print(f"Saved MD: {out_md}")
    print(f"Saved plots in: {save_dir}")

    if baseline:
        print(f"Baseline acc1/acc5: {baseline['acc1']:.4f}/{baseline['acc5']:.4f}")

    if summaries:
        best = max((s for s in summaries if s["best_test_acc1"] is not None), key=lambda s: s["best_test_acc1"], default=None)
        if best:
            print(f"Best experiment by acc1: {best['experiment']} ({best['best_test_acc1']:.4f})")


if __name__ == "__main__":
    main()

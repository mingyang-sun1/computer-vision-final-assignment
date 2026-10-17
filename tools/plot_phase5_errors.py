"""Render Phase 5 figures from audited CSVs, with optional local photo examples.

    python tools/plot_phase5_errors.py --analysis-dir docs/evidence/phase5_run_20261017

Numerical plots need matplotlib. Photo panels also need Pillow and the original
course training photos. This script never runs a model or creates image content.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


LABELS = {
    "best_absolute": "Closest predictions",
    "largest_overprediction": "Largest overestimates",
    "largest_underprediction": "Largest underestimates",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def new_directory(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise ValueError(f"Output directory already exists: {path}. Choose a new name.")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save_figure(fig: Any, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight", metadata={"Date": None})
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(directory / f"{stem}.png", bbox_inches="tight", dpi=180)


def draw_diagnostics(analysis_dir: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    report = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
    buckets = read_rows(analysis_dir / "price_buckets.csv")
    rows = read_rows(analysis_dir / "per_image_errors.csv")
    if not buckets or not rows:
        raise ValueError("Analysis tables are empty.")
    output_dir = new_directory(output_dir)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "svg.fonttype": "none"})
    blue, orange, red = "#246b8e", "#b87025", "#ad3f49"
    x = list(range(1, len(buckets) + 1))
    deciles = report.get("bucket_method", {}).get("requested_count") == 10 and len(buckets) == 10
    group_name = "decile" if deciles else "quantile group"
    names = [row["bucket"].replace("B", "D") if deciles else row["bucket"] for row in buckets]
    values = lambda key: [float(row[key]) for row in buckets]
    title = f"E5 ensemble: error by true-price {group_name} (n = {len(rows):,})"
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.3), constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold")
    ax = axes[0, 0]
    ax.bar(x, values("mae"), color=blue, width=.72)
    ax.set(title="A. Mean absolute error", ylabel="MAE (USD 1,000)")
    ax = axes[0, 1]
    biases = values("bias")
    ax.bar(x, biases, color=[orange if b >= 0 else blue for b in biases], width=.72)
    ax.axhline(0, color="#555555", linewidth=.8)
    ax.set(title="B. Signed bias: predicted - true", ylabel="Mean residual (USD 1,000)")
    ax = axes[1, 0]
    shares = values("sse_share")
    bars = ax.bar(x, shares, color=[blue] * (len(x) - 1) + [red], width=.72)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    for bar, value in zip(bars, shares):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .012, f"{value:.1%}",
                ha="center", va="bottom", fontsize=8)
    ax.set(title="C. Contribution to total squared error", ylabel="Share of total SSE",
           ylim=(0, max(shares) * 1.22))
    ax = axes[1, 1]
    ax.plot(x, values("mean_true"), "o-", label="Mean true price", color="#444444")
    ax.plot(x, values("mean_pred"), "s-", label="Mean predicted price", color=blue)
    ax.set(title="D. Mean true and predicted prices", ylabel="Price (USD 1,000)")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes.flat:
        ax.set_xticks(x, names)
        ax.set_xlabel(f"True-price {group_name} (lowest to highest)")
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    save_figure(fig, output_dir, "phase5_price_buckets")
    plt.close(fig)

    true = [float(row["price_true"]) for row in rows]
    pred = [float(row["price_pred"]) for row in rows]
    residual = [float(row["residual"]) for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), constrained_layout=True)
    fig.suptitle("E5 ensemble: validation predictions and residuals", fontsize=15,
                 fontweight="bold")
    ax = axes[0]
    ax.scatter(true, pred, s=13, alpha=.36, color=blue, edgecolors="none")
    limit = max(max(true), max(pred)) * 1.04
    ax.plot([0, limit], [0, limit], "--", color="#555555", linewidth=1,
            label="Perfect prediction")
    ax.set(xlim=(0, limit), ylim=(0, limit), xlabel="True price (USD 1,000)",
           ylabel="Predicted price (USD 1,000)", title="A. Predicted versus true price")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax = axes[1]
    ax.hist(residual, bins=30, color=blue, alpha=.88, edgecolor="white", linewidth=.5)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=1, label="Zero error")
    ax.axvline(float(report["overall"]["bias"]), color=orange, linewidth=1.5,
               label=f"Mean = {float(report['overall']['bias']):+.1f}")
    ax.set(xlabel="Residual: predicted - true (USD 1,000)", ylabel="Image count",
           title="B. Residual distribution")
    ax.legend(frameon=False, fontsize=9)
    for ax in axes:
        ax.grid(alpha=.16)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:,.0f}"))
    save_figure(fig, output_dir, "phase5_prediction_diagnostics")
    plt.close(fig)
    metadata = {"analysis_source": analysis_dir.name, "images_read": False,
                "source_files": [{"name": name,
                  "sha256": hashlib.sha256((analysis_dir / name).read_bytes()).hexdigest()}
                 for name in ("summary.json", "price_buckets.csv", "per_image_errors.csv")],
                "figures": ["phase5_price_buckets", "phase5_prediction_diagnostics"],
                "formats": ["svg", "pdf", "png"]}
    (output_dir / "figure_sources.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                    encoding="utf-8")
    print(f"Numerical figures: {output_dir}")


def draw_photo_examples(analysis_dir: Path, image_root: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps

    rows = read_rows(analysis_dir / "representative_examples.csv")
    if not rows:
        raise ValueError("No representative examples were selected.")
    image_root = image_root.resolve(strict=True)
    images, hashes = {}, {}
    # Validate every required original before creating an output directory.
    for row in rows:
        ident = row["imageid"]
        if Path(ident).name != ident or "/" in ident or "\\" in ident:
            raise ValueError(f"Expected a plain image filename: {ident}")
        path = (image_root / ident).resolve(strict=True)
        if not path.is_relative_to(image_root):
            raise ValueError(f"Image is outside the supplied image root: {ident}")
        with Image.open(path) as original:
            preview = ImageOps.exif_transpose(original).convert("RGB")
            preview.thumbnail((1000, 700))
            images[ident] = preview.copy()
        hashes[ident] = hashlib.sha256(path.read_bytes()).hexdigest()
    output_dir = new_directory(output_dir)
    fig, axes = plt.subplots(len(LABELS), 3, figsize=(10.8, 10.6), squeeze=False,
                             constrained_layout=True)
    fig.suptitle("E5 validation examples selected by numerical error", fontsize=15,
                 fontweight="bold")
    for group_index, (category, label) in enumerate(LABELS.items()):
        selected = sorted((row for row in rows if row["category"] == category),
                          key=lambda row: int(row["rank"]))
        for col, ax in enumerate(axes[group_index]):
            ax.axis("off")
            if col >= len(selected):
                continue
            row = selected[col]
            ax.imshow(images[row["imageid"]])
            ax.set_title(f"{label} #{row['rank']} | {row['imageid']}\n"
                         f"True {float(row['price_true']):,.1f}  /  Pred {float(row['price_pred']):,.1f}\n"
                         f"Residual {float(row['residual']):+,.1f} (USD 1,000)", fontsize=9)
    save_figure(fig, output_dir, "phase5_selected_examples")
    plt.close(fig)
    fields = ["category", "rank", "imageid", "price_true", "price_pred", "residual",
              "image_sha256", "visual_review_status", "visible_features", "possible_explanation"]
    with (output_dir / "visual_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            record = {key: row[key] for key in fields if key in row}
            record.update(image_sha256=hashes[row["imageid"]], visual_review_status="NOT_REVIEWED",
                          visible_features="", possible_explanation="")
            writer.writerow(record)
    (output_dir / "README.txt").write_text(
        "Prices and errors are in USD 1,000. Examples are selected extremes, not a random sample.\n"
        "The photo panel does not establish a cause of prediction error.\n"
        "Inspect the actual photos and complete visual_review.csv before writing visual claims.\n"
        "Use the panel in the written report; keep course photos out of the code submission.\n",
        encoding="utf-8")
    print(f"Photo examples: {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, help="New directory for numerical plots")
    parser.add_argument("--image-root", type=Path,
                        help="Original course training-photo directory, usually CV_DATA_ROOT/train")
    parser.add_argument("--image-output-dir", type=Path,
                        help="New local directory for photo panels and review CSV")
    parser.add_argument("--examples-only", action="store_true",
                        help="Generate local photo panels without regenerating numerical plots")
    args = parser.parse_args()
    analysis_dir = args.analysis_dir.resolve()
    if args.examples_only and args.image_root is None:
        parser.error("--examples-only requires --image-root")
    try:
        if not args.examples_only:
            draw_diagnostics(analysis_dir, args.output_dir or analysis_dir / "figures")
        if args.image_root is not None:
            repo_root = Path(__file__).resolve().parents[1]
            draw_photo_examples(analysis_dir, args.image_root,
                                args.image_output_dir or repo_root / "outputs" / "phase5_examples")
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"Phase 5 figure generation failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

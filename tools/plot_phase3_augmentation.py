"""Create a static E2/E4 comparison figure from a successful CSV audit.

Requires matplotlib from requirements.txt. This script reads an existing
audit; it does not train a model, load images, or invent training curves.
Existing figure files are never overwritten.
"""
from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path


def read_metrics(path: Path) -> tuple[list[dict], float]:
    """Accept only a complete, successful audit of both prediction files."""
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")

    report = json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
    if not isinstance(report, dict) or not isinstance(report.get("comparison"), dict):
        raise ValueError("The audit does not contain a valid E2/E4 comparison")
    summary = report.get("summary")
    if not isinstance(summary, dict) or summary.get("fail") != 0:
        raise ValueError("Resolve failed audit checks before creating a report figure")
    runs = report.get("runs")
    if not isinstance(runs, dict):
        raise ValueError("The audit does not contain run records")
    rows = []
    for tag in ("E2_finetune", "E4_augment"):
        run = runs.get(tag, {})
        if not isinstance(run, dict) or run.get("row_count") != 1000:
            raise ValueError(f"{tag}: expected the fixed 1,000-row validation set")
        metrics = run.get("metrics", {})
        if not isinstance(metrics, dict):
            raise ValueError(f"{tag}: missing metric record")
        for key in ("mse", "mae", "r2"):
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{tag}: missing or non-finite {key}")
        if metrics["mse"] <= 0 or metrics["mae"] < 0:
            raise ValueError(f"{tag}: invalid error metrics")
        rows.append(metrics)
    relative = 100.0 * (rows[1]["mse"] / rows[0]["mse"] - 1.0)
    recorded = report["comparison"].get("mse_change_percent")
    if type(recorded) not in (int, float) or not math.isclose(relative, recorded, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("The comparison percentage disagrees with the run metrics")
    return rows, relative


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args(argv)
    try:
        rows, relative = read_metrics(args.audit)
        destinations = [args.out_dir / f"phase3_augmentation_comparison.{extension}"
                        for extension in ("svg", "pdf")]
        for path in destinations:
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite {path}; choose a new --out-dir")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter

        with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                             "svg.fonttype": "none", "pdf.fonttype": 42,
                             "axes.spines.top": False, "axes.spines.right": False}):
            figure, axes = plt.subplots(1, 2, figsize=(8.6, 4.7))
            figure.subplots_adjust(left=0.105, right=0.97, bottom=0.23,
                                   top=0.72, wspace=0.45)
            figure.suptitle("Validation results with and without augmentation",
                            y=0.96, fontsize=14, fontweight="bold")
            figure.text(0.5, 0.875,
                        "Same 1,000 validation images | Metrics from committed rounded predictions",
                        ha="center", fontsize=9, color="#444444")
            figure.text(0.5, 0.80, f"E4 minus E2: {relative:+.2f}% MSE",
                        ha="center", fontsize=12, fontweight="bold")
            for axis, key, title, unit in zip(
                axes, ("mse", "mae"), ("Mean squared error", "Mean absolute error"),
                ("(USD 1,000) squared", "USD 1,000")
            ):
                values = [row[key] for row in rows]
                bars = axis.bar([0, 1], values, width=0.55,
                                color=["#dddddd", "#787878"], edgecolor="#333333",
                                linewidth=0.8, zorder=3)
                bars[1].set_hatch("///")
                axis.set_xticks([0, 1], ["E2\nNo augmentation", "E4\nAugmentation"])
                axis.set_ylabel(unit)
                axis.set_title(title, fontsize=11, pad=12)
                axis.set_ylim(0, max(values) * 1.23)
                axis.set_axisbelow(True)
                axis.grid(axis="y", color="#e1e1e1", linewidth=0.6)
                axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
                for bar, value in zip(bars, values):
                    label = f"{value:,.1f}" if key == "mse" else f"{value:,.2f}"
                    axis.text(bar.get_x() + bar.get_width() / 2,
                              value + max(values) * 0.045, label,
                              ha="center", va="bottom", fontweight="bold", fontsize=10)
            figure.text(0.5, 0.06,
                        "Lower is better. One stored run per setting; this figure is not a training reproduction.",
                        ha="center", fontsize=8.5, color="#444444")
            buffers = []
            for extension in ("svg", "pdf"):
                buffer = io.BytesIO()
                figure.savefig(buffer, format=extension, facecolor="white",
                               metadata={"Creator": "Phase 3 augmentation audit"})
                buffers.append(buffer.getvalue())
            plt.close(figure)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        for path, payload in zip(destinations, buffers):
            with path.open("xb") as stream:
                stream.write(payload)
            print(f"Created {path}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, ImportError) as error:
        print(f"ERROR: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

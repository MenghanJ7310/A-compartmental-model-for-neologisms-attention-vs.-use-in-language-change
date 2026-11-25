#!/usr/bin/env python3
"""Reconstructed score calculation, 2026-11-21; NOT the original model fitter.

Recomputes scores from saved per-word R2 and test RMSE, without refitting models.
The 70/30 training/test split is upstream of this calculation and is not a
score weight. The defaults below reproduce the saved 176-word score table.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREDICTION_WEIGHT = 0.50
CUMULATIVE_FIT_WEIGHT = 0.50
CUMULATIVE_R2_REFERENCE = 0.80


def logisticness(r2, rmse_logistic, rmse_arima, rmse_naive,
                 prediction_weight=PREDICTION_WEIGHT,
                 fit_weight=CUMULATIVE_FIT_WEIGHT):
    values = [r2, rmse_logistic, rmse_arima, rmse_naive,
              prediction_weight, fit_weight]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("All inputs must be finite; missing metrics are not replaced by zero.")
    if min(rmse_logistic, rmse_arima, rmse_naive) < 0:
        raise ValueError("RMSE cannot be negative.")
    if min(prediction_weight, fit_weight) < 0 or not math.isclose(
            prediction_weight + fit_weight, 1.0, abs_tol=1e-12):
        raise ValueError("Nonnegative score weights must sum to one.")
    baseline = min(rmse_arima, rmse_naive)
    # Eight saved rows have baseline RMSE == 0 and a saved prediction gain of 0.
    prediction_gain = max(0.0, 1.0 - rmse_logistic / baseline) if baseline > 0 else 0.0
    cumulative_fit = min(1.0, max(0.0, (r2 - CUMULATIVE_R2_REFERENCE)
                                / (1.0 - CUMULATIVE_R2_REFERENCE)))
    score = prediction_weight * prediction_gain + fit_weight * cumulative_fit
    return baseline, prediction_gain, cumulative_fit, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/v3lite_logisticness_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records or len({r["word"] for r in records}) != len(records):
        raise ValueError("Expected nonempty data with unique word identifiers.")
    checks, zero_baseline = [], []
    for row in records:
        b, p, f, score = logisticness(*[float(row[k]) for k in [
            "r2_cumulative_logistic", "rmse_test_logistic_inc",
            "rmse_test_arima", "rmse_test_naive"]])
        checks.append({
            "word": row["word"], "n_points_saved": int(row["n_points"]),
            "baseline_rmse_recomputed": b, "prediction_gain_recomputed": p,
            "cumulative_fit_recomputed": f, "score_recomputed": score,
            "score_saved": float(row["logisticness_score"]),
            "baseline_absolute_error": abs(b - float(row["rmse_test_baseline_min"])),
            "prediction_absolute_error": abs(p - float(row["component_pred_gain"])),
            "fit_absolute_error": abs(f - float(row["component_cum_pref"])),
            "score_absolute_error": abs(score - float(row["logisticness_score"])),
        })
        if b == 0:
            zero_baseline.append(row["word"])
    errors = {k: max(r[k] for r in checks) for k in checks[0] if k.endswith("absolute_error")}
    summary = {
        "scope": "Score recalculation from saved metrics; no model fitting or train/test split performed.",
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "n_words": len(checks), "prediction_weight": PREDICTION_WEIGHT,
        "cumulative_fit_weight": CUMULATIVE_FIT_WEIGHT,
        "cumulative_r2_reference": CUMULATIVE_R2_REFERENCE,
        "median_recomputed": statistics.median(r["score_recomputed"] for r in checks),
        "max_absolute_errors": errors,
        "all_metrics_match_within_1e_minus_12": max(errors.values()) < 1e-12,
        "zero_baseline_words": zero_baseline,
        "n_zero_prediction_gain": sum(r["prediction_gain_recomputed"] == 0 for r in checks),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "score_recalculation_176.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    (args.output_dir / "score_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["all_metrics_match_within_1e_minus_12"]:
        raise SystemExit("Saved scores do not match the reconstructed calculation.")


if __name__ == "__main__":
    main()

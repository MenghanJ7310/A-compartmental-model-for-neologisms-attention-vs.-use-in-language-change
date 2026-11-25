#!/usr/bin/env python3
"""Audit current raw-file record counts against the saved 176-word score set.

Thresholds 8 and 12 are DIAGNOSTIC hypotheses, not recovered historical rules.
This script does not fit models, change raw files, or reconstruct missing
181-word results. A valid observation here has a parseable date and a finite,
nonnegative count. Record counts are NOT necessarily calendar-month counts.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower().replace("_result", "").replace("-time", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/monitor_raw")
    parser.add_argument("--scores", type=Path, default=ROOT / "data/v3lite_logisticness_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    scores = pd.read_csv(args.scores, encoding="utf-8-sig")
    scores["key"] = scores["word"].map(canonical)
    if scores["key"].duplicated().any():
        raise ValueError("Duplicate canonical word identifiers in score file.")
    score_n = scores.set_index("key")["n_points"].to_dict()
    rows = []
    for path in sorted(args.raw_dir.glob("*.csv")):
        raw = pd.read_csv(path, encoding="utf-8-sig")
        cols = {str(c).strip().lower(): c for c in raw.columns}
        time_col = next((cols[c] for c in ["time", "date"] if c in cols), None)
        count_col = cols.get("count")
        if time_col is None or count_col is None:
            raise ValueError(f"{path.name}: explicit date/count columns not found.")
        dates = pd.to_datetime(raw[time_col], errors="coerce", format="mixed")
        counts = pd.to_numeric(raw[count_col], errors="coerce")
        valid = dates.notna() & np.isfinite(counts) & counts.ge(0)
        n = int(valid.sum())
        key = canonical(path.stem)
        saved = score_n.get(key)
        rows.append({
            "word": path.stem.removesuffix("_result"), "file": path.name, "key": key,
            "n_raw_rows": len(raw), "n_valid_dated_records": n,
            "n_invalid_date_rows": int(dates.isna().sum()),
            "n_invalid_count_rows": int((~np.isfinite(counts) | counts.lt(0)).sum()),
            "n_unique_valid_counts": int(counts[valid].nunique()),
            "sum_valid_counts": float(counts[valid].sum()),
            "present_in_saved_score_table": saved is not None,
            "n_points_in_saved_score_table": saved,
            "matches_saved_n_points": n == saved if saved is not None else None,
            "meets_diagnostic_n_ge_8": n >= 8,
            "meets_diagnostic_n_ge_12": n >= 12,
            "historical_exclusion_reason": "not established from recovered records",
        })
    data = pd.DataFrame(rows)
    if data.empty or data["key"].duplicated().any():
        raise ValueError("Raw-file inventory is empty or contains identifier collisions.")
    keys_8 = set(data.loc[data.meets_diagnostic_n_ge_8, "key"])
    keys_12 = set(data.loc[data.meets_diagnostic_n_ge_12, "key"])
    saved_keys = set(score_n)
    candidate = data[data["key"].isin(keys_8 - keys_12)].copy()
    candidate["historical_exclusion_reason"] = (
        "Candidate only: 8-11 valid records and absent from saved 176-word table; "
        "membership in the original 181-word comparison is unverified.")
    summary = {
        "n_raw_files": len(data), "n_saved_score_words": len(saved_keys),
        "diagnostic_n_ge_8": len(keys_8), "diagnostic_n_ge_12": len(keys_12),
        "saved_score_word_set_equals_n_ge_12": saved_keys == keys_12,
        "saved_words_missing_raw_files": sorted(saved_keys - set(data["key"])),
        "candidate_difference_words": candidate["word"].tolist(),
        "saved_n_points_mismatches": data[
            data.present_in_saved_score_table & data.matches_saved_n_points.eq(False)
        ][["word", "n_valid_dated_records", "n_points_in_saved_score_table"]].to_dict("records"),
        "constant_series_present_in_score_table": data[
            data.present_in_saved_score_table & data.n_unique_valid_counts.eq(1)
        ]["word"].tolist(),
        "interpretation": (
            "The 8/12 record thresholds reproduce the reported sample sizes and "
            "the saved 176-word membership, but this does not establish the "
            "original filtering rule or the identity of the original 181-word set."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.drop(columns="key").to_csv(args.output_dir / "sample_inventory_190.csv", index=False, encoding="utf-8-sig")
    candidate.drop(columns="key").to_csv(args.output_dir / "candidate_five_words.csv", index=False, encoding="utf-8-sig")
    data[~data.present_in_saved_score_table].drop(columns="key").to_csv(
        args.output_dir / "words_absent_from_score_table_14.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "sample_size_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

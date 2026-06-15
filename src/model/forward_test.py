"""
forward_test.py (model) — the honest production-value test: freeze today's scores,
then measure precision@K on GENUINELY-FUTURE exclusions when the next LEIE/AHCCCS/NV
download arrives. A backtest is in-sample-optimistic; this is the real thing —
"score today, see who actually gets excluded next."

It also settles the open label question the backtests could not: it freezes BOTH the
production (all-LEIE) and the fraud_positive scores, so the future download tells us
which model better predicts NEW exclusions.

Two modes:

  freeze   — snapshot, per NPI: both model scores, whether the provider is excluded
             TODAY (on LEIE/AHCCCS/NV as of this baseline), net_paid, reliability.
             Records the baseline date and the source LEIE file date. Run once now.

  measure  — given a FUTURE exclusion download (a re-pulled Caught.csv, and optionally
             refreshed AHCCCS/NV), restrict to providers NOT excluded at baseline,
             label those newly excluded in the future pull, and report precision@K +
             lift for each model on that genuinely-out-of-sample set. Run later.

Run now:
    python -m src.model.forward_test freeze
Run when the next LEIE drops (e.g. 2026-09):
    python -m src.model.forward_test measure --future-leie /path/to/Caught.csv \
        [--future-ahcccs <pdf>] [--future-nv <pdf>] [--baseline-date 2026-06-15]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .data import log, require

OUT_DIR = config.MODEL_DATA_DIR / "forward_test"
PROD_SCORES = config.SCORES_DIR / "provider_model_scores.parquet"
FRAUD_SCORES = config.MODEL_DATA_DIR / "fraud_positive" / "npi_scores_fraud.parquet"
LABELS = config.MODEL_DATA_DIR / "labels" / "expanded_labels.parquet"
FRAUD_RELEVANT = {"1128a1", "1128a2", "1128a3", "1128b1", "1128b2", "1128b3", "1128b7"}


def leie_npis(caught_csv):
    """(all-LEIE npis, fraud-relevant-LEIE npis) from a LEIE Caught.csv."""
    df = pd.read_csv(caught_csv, dtype=str, keep_default_na=False)
    df = df[df["NPI"].str.fullmatch(r"[12]\d{9}")]
    return set(df["NPI"]), set(df.loc[df["EXCLTYPE"].isin(FRAUD_RELEVANT), "NPI"])


def freeze(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("[freeze] loading both models' scores + today's exclusion membership")
    prod = pd.read_parquet(PROD_SCORES, columns=["npi", "model_score", "net_paid",
                                                 "not_scored", "org_legal_name"])
    prod = prod.rename(columns={"model_score": "score_leie"})
    fr = pd.read_parquet(FRAUD_SCORES, columns=["npi", "model_score"]).rename(
        columns={"model_score": "score_fraud"})
    lab = pd.read_parquet(LABELS, columns=["npi", "on_leie_any", "on_leie_fraud",
                                           "on_ahcccs", "on_nv", "fraud_positive"])
    base = prod.merge(fr, on="npi", validate="1:1").merge(lab, on="npi", how="left", validate="1:1")
    base["excluded_today"] = (base[["on_leie_any", "on_ahcccs", "on_nv"]]
                              .fillna(False).any(axis=1))
    require("both score columns present and non-null",
            bool(base["score_leie"].notna().all() and base["score_fraud"].notna().all()))

    stamp = args.baseline_date
    out = OUT_DIR / f"baseline_{stamp}.parquet"
    base.to_parquet(out, index=False)
    leie_date = pd.Timestamp(Path(config.DATA_DIR / "preclean" / "Caught.csv").stat().st_mtime,
                             unit="s").date().isoformat()
    meta = {"baseline_date": stamp, "leie_file_mtime": leie_date,
            "n_npis": len(base), "n_excluded_today": int(base["excluded_today"].sum()),
            "models": ["score_leie (production all-LEIE)", "score_fraud (expanded fraud_positive)"],
            "how_to_measure": "re-pull Caught.csv in N months, then: "
                              "python -m src.model.forward_test measure --future-leie <path>"}
    (OUT_DIR / f"baseline_{stamp}_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"    froze {len(base):,} NPIs | excluded today {meta['n_excluded_today']:,} "
        f"| LEIE file {leie_date}")
    log(f"    → {out}\n    Re-run `measure --future-leie <new Caught.csv>` when the next LEIE drops.")


def precision_at_k(y, s, ks=(50, 100, 500, 1000)):
    order = np.argsort(-s)
    yr = y[order]
    base = y.mean()
    out = {}
    for k in ks:
        if k <= len(yr):
            p = yr[:k].sum() / k
            out[f"p@{k}"] = float(p)
            out[f"lift@{k}"] = float(p / base) if base > 0 else float("nan")
    return out


def measure(args):
    bl = OUT_DIR / f"baseline_{args.baseline_date}.parquet"
    require("baseline snapshot exists — run `freeze` first", bl.exists(), str(bl))
    base = pd.read_parquet(bl)
    log(f"[measure] baseline {args.baseline_date}: {len(base):,} NPIs")

    f_all, f_fraud = leie_npis(args.future_leie)
    future_excl = set(f_all)
    for pdf in (args.future_ahcccs, args.future_nv):     # optional refreshed state lists
        if pdf:
            import re
            from pypdf import PdfReader
            text = "\n".join((pg.extract_text() or "") for pg in PdfReader(pdf).pages)
            future_excl |= set(re.findall(r"\b[12]\d{9}\b", text))
    # candidates = NOT excluded at baseline (the genuinely out-of-sample population)
    cand = base[~base["excluded_today"]].copy()
    cand["newly_excluded"] = cand["npi"].isin(future_excl)
    cand["newly_excluded_fraud"] = cand["npi"].isin(f_fraud)
    n_new = int(cand["newly_excluded"].sum())
    require("the future pull contains NEW exclusions among baseline-clean providers",
            n_new > 0, f"{n_new} new — is this actually a later LEIE pull?")
    log(f"    {len(cand):,} baseline-clean candidates | {n_new} newly excluded in the future pull")

    rows = []
    for model, col in (("production (all-LEIE)", "score_leie"),
                       ("expanded (fraud_positive)", "score_fraud")):
        for target, tcol in (("any new LEIE", "newly_excluded"),
                             ("new fraud-relevant LEIE", "newly_excluded_fraud")):
            y = cand[tcol].to_numpy().astype(int)
            if y.sum() == 0:
                continue
            m = precision_at_k(cand[col].to_numpy(), y)
            rows.append({"model": model, "target": target, "n_pos": int(y.sum()),
                         "base_rate": float(y.mean()), **m})
    res = pd.DataFrame(rows)
    out = OUT_DIR / f"forward_result_{args.baseline_date}.csv"
    res.to_csv(out, index=False)
    log("\nFORWARD-TEST RESULT (precision@K / lift on genuinely-future exclusions)")
    log(res.to_string(index=False))
    log(f"\n    → {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    pf = sub.add_parser("freeze"); pf.add_argument("--baseline-date", default="2026-06-15")
    pm = sub.add_parser("measure")
    pm.add_argument("--future-leie", required=True)
    pm.add_argument("--future-ahcccs", default=None)
    pm.add_argument("--future-nv", default=None)
    pm.add_argument("--baseline-date", default="2026-06-15")
    args = p.parse_args()
    (freeze if args.cmd == "freeze" else measure)(args)


if __name__ == "__main__":
    main()

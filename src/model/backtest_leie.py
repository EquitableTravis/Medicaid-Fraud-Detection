"""
backtest_leie.py (model) — the supervised model's score through the SAME LEIE
backtest as the unsupervised anomaly score (src/backtest/backtest_leie.py),
plus a head-to-head on the identical company universe.

Methodology mirrored exactly: label every company by fraud-relevant OIG LEIE
exclusion (NPI match first, name+state fallback), decile lift vs a
billing-size-only baseline, within-billing-quartile stratification,
permutation test, bootstrap CI.

ONE THING THE ANOMALY BACKTEST DID NOT NEED: the model TRAINED on LEIE labels
(578 positive NPIs; 462 in the train split, 116 in the early-stopping val
split), so companies containing trained-on NPIs are partly memorized. Every
headline is therefore reported twice:
  - "all"      — every hit (apples-to-apples with the anomaly backtest's
                 evaluation set, but optimistic for the model);
  - "held-out" — train-split-NPI companies REMOVED from the universe, so the
                 only hits left are val-split NPIs and name+state matches the
                 model never saw.
The head-to-head runs both scores on the intersection universe (companies
scored by BOTH methods) with identical labels — that is the decisive table.

Writes MODEL_BACKTEST_REPORT.md + model_backtest.json to
~/Desktop/Data/Model/backtest/.

Run (after score.py; needs src/backtest/'s company_scores_full.parquet):
    python -m src.model.backtest_leie
"""

import json

import numpy as np
import pandas as pd

from ..backtest.backtest_leie import (FRAUD_RELEVANT, LEIE, NPI_RE, decile,
                                      norm_name, states_set, timing_bucket)
from . import config
from .data import log, require

OUT_DIR = config.MODEL_DATA_DIR / "backtest"
ANOMALY_FULL = config.find_input("company_scores_full.parquet")
SEED = 0
N_RESAMPLE = 1000


def load_leie_labels():
    leie = pd.read_csv(LEIE, dtype=str, keep_default_na=False)
    fr = leie[leie["EXCLTYPE"].isin(FRAUD_RELEVANT)].copy()
    has_npi = fr["NPI"].str.match(r"^\d{10}$") & (fr["NPI"] != "0000000000")
    npi_lab = (fr[has_npi].groupby("NPI")
               .agg(et=("EXCLTYPE", lambda s: ";".join(sorted(set(s)))),
                    ed=("EXCLDATE", "min")))
    npi_map = {i: (r.et, r.ed) for i, r in npi_lab.iterrows()}
    name_map = {}
    for r in fr[(~has_npi) & (fr["BUSNAME"].str.strip() != "")].itertuples():
        name_map.setdefault((norm_name(r.BUSNAME), r.STATE.upper()), []).append(
            (r.EXCLTYPE, r.EXCLDATE))
    return npi_map, name_map


def label_universe(u, npi_map, name_map):
    def label(row):
        hit_npi, ets, eds = [], [], []
        for n in NPI_RE.findall(str(row.npi_list)):
            if n in npi_map:
                hit_npi.append(n)
                ets.append(npi_map[n][0])
                eds.append(npi_map[n][1])
        conf = "npi" if hit_npi else ""
        if not hit_npi:
            nk = norm_name(row.company_name)
            for st in states_set(row.states):
                if (nk, st) in name_map:
                    for et, ed in name_map[(nk, st)]:
                        ets.append(et)
                        eds.append(ed)
                    conf = "name"
        hit = 1 if conf else 0
        ed = min([e for e in eds if e and e != "00000000"], default="")
        return pd.Series({"hit": hit, "matched_npi": ";".join(hit_npi),
                          "match_confidence": conf,
                          "timing_bucket": timing_bucket(ed) if hit else ""})
    return pd.concat([u, u.apply(label, axis=1)], axis=1)


def lift_metrics(u, score_col, rng):
    """Top-decile lift vs billing baseline, within-quartile, permutation, bootstrap."""
    u = u.copy()
    base = u["hit"].mean()
    u["score_decile"] = decile(u[score_col])
    u["billing_decile"] = decile(u["company_net_paid"])
    top = u.loc[u["score_decile"] == 10, "hit"].mean() / base
    bill_top = u.loc[u["billing_decile"] == 10, "hit"].mean() / base
    within = {}
    u["size_band"] = pd.qcut(u["company_net_paid"].rank(method="first"), 4,
                             labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = u[u["size_band"] == q]
        qbase = sub["hit"].mean()
        qtop = sub.loc[decile(sub[score_col]) == 10, "hit"].mean()
        within[q] = round(float(qtop / qbase), 2) if qbase > 0 else None
    sc, h = u[score_col].to_numpy(), u["hit"].to_numpy()
    n, n_hit = len(u), int(h.sum())
    obs = sc[h == 1].sum()
    perm = np.array([sc[rng.choice(n, n_hit, replace=False)].sum()
                     for _ in range(N_RESAMPLE)])
    p_val = float((np.sum(perm >= obs) + 1) / (N_RESAMPLE + 1))
    d10 = (u["score_decile"] == 10).to_numpy()
    boot = np.empty(N_RESAMPLE)
    for i in range(N_RESAMPLE):
        s = rng.choice(n, n, replace=True)
        b = h[s].mean()
        boot[i] = (h[s][d10[s]].mean() / b) if b > 0 and d10[s].any() else np.nan
    ci = [round(float(np.nanpercentile(boot, q)), 2) for q in (2.5, 97.5)]
    return {"n": n, "hits": n_hit, "base_rate": round(float(base), 6),
            "top_decile_lift": round(float(top), 2),
            "billing_top_decile_lift": round(float(bill_top), 2),
            "within_size_top_decile_lift": within,
            "permutation_p": round(p_val, 4), "bootstrap_ci95": ci}


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("[1/5] Loading model + anomaly company scores, building universe")
    comp = pd.read_parquet(config.SCORES_DIR / "company_model_scores.parquet")
    roll = pd.read_parquet(config.COMPANY_ROLLUP,
                           columns=["company_id", "states", "npi_list"])
    u = comp.merge(roll, on="company_id", how="left", validate="1:1")
    u = u[u["company_model_score_max"].notna()].reset_index(drop=True)
    anom = pd.read_parquet(ANOMALY_FULL,
                           columns=["company_id", "company_anomaly_score"])
    u = u.merge(anom, on="company_id", how="left", validate="1:1")
    log(f"    model-scored universe: {len(u):,} companies")

    log("[2/5] Labelling fraud-relevant LEIE hits (NPI, then name+state)")
    npi_map, name_map = load_leie_labels()
    u = label_universe(u, npi_map, name_map)
    log(f"    hits: {int(u['hit'].sum())} "
        f"({int((u['match_confidence']=='npi').sum())} npi / "
        f"{int((u['match_confidence']=='name').sum())} name)")

    log("[3/5] Splitting hits by training exposure (model saw 578 LEIE NPIs)")
    pu_pos = pd.read_parquet(config.PU_TRAINING_PARQUET,
                             columns=["npi", config.LABEL])
    pos_npis = set(pu_pos.loc[pu_pos[config.LABEL], "npi"])
    val = pd.read_parquet(config.ARTIFACTS_DIR / "val_predictions.parquet")
    val_pos = set(val.loc[val["y_true"] == 1, "npi"])
    train_pos = pos_npis - val_pos
    require("PU positive split adds up", len(train_pos) + len(val_pos) == len(pos_npis))
    contains_train = u["npi_list"].map(
        lambda s: any(n in train_pos for n in NPI_RE.findall(str(s))))
    held = u[~contains_train].reset_index(drop=True)
    log(f"    held-out universe: {len(held):,} companies, {int(held['hit'].sum())} hits "
        f"(train-split-NPI companies removed: {int(contains_train.sum()):,})")

    log("[4/5] Lift metrics: model (all + held-out) and head-to-head intersection")
    results = {
        "model_all": lift_metrics(u, "company_model_score_max", rng),
        "model_held_out": lift_metrics(held, "company_model_score_max", rng),
    }
    both = u[u["company_anomaly_score"].notna()].reset_index(drop=True)
    both_held = both[~both.index.isin(
        both.index[both["npi_list"].map(
            lambda s: any(n in train_pos for n in NPI_RE.findall(str(s))))])]
    results["head_to_head_universe"] = len(both)
    for name, frame in [("h2h_all", both), ("h2h_held_out", both_held.reset_index(drop=True))]:
        results[name] = {
            "model": lift_metrics(frame, "company_model_score_max", rng),
            "anomaly": lift_metrics(frame, "company_anomaly_score", rng),
        }

    log("[5/5] Writing report")
    (OUT_DIR / "model_backtest.json").write_text(json.dumps(results, indent=2))
    h_a, h_h = results["h2h_all"], results["h2h_held_out"]
    (OUT_DIR / "MODEL_BACKTEST_REPORT.md").write_text(f"""# MODEL vs ANOMALY — LEIE backtest, identical methodology

Universe = companies scored by BOTH methods: {results['head_to_head_universe']:,}
({h_a['model']['hits']} fraud-relevant LEIE hits; base rate {h_a['model']['base_rate']:.4%}).
"Held-out" removes every company containing a train-split LEIE NPI, leaving
{h_h['model']['hits']} hits the model never trained on (val-split NPIs + name matches).

## Head-to-head: top-decile lift (vs billing baseline {h_a['model']['billing_top_decile_lift']}x)

| eval set | model | anomaly |
|---|---|---|
| all hits | **{h_a['model']['top_decile_lift']}x** (p={h_a['model']['permutation_p']}, CI {h_a['model']['bootstrap_ci95']}) | {h_a['anomaly']['top_decile_lift']}x (p={h_a['anomaly']['permutation_p']}, CI {h_a['anomaly']['bootstrap_ci95']}) |
| held-out only | **{h_h['model']['top_decile_lift']}x** (p={h_h['model']['permutation_p']}, CI {h_h['model']['bootstrap_ci95']}) | {h_h['anomaly']['top_decile_lift']}x (p={h_h['anomaly']['permutation_p']}, CI {h_h['anomaly']['bootstrap_ci95']}) |

Within-billing-quartile top-decile lift —
model (held-out): {h_h['model']['within_size_top_decile_lift']} |
anomaly (held-out): {h_h['anomaly']['within_size_top_decile_lift']}

## Model on its own full universe ({results['model_all']['n']:,} companies)

| eval set | top-decile lift | p | CI95 |
|---|---|---|---|
| all hits | {results['model_all']['top_decile_lift']}x | {results['model_all']['permutation_p']} | {results['model_all']['bootstrap_ci95']} |
| held-out | {results['model_held_out']['top_decile_lift']}x | {results['model_held_out']['permutation_p']} | {results['model_held_out']['bootstrap_ci95']} |

## Caveats
- The model trained on LEIE: "all hits" rows are optimistic (memorization);
  judge it on the held-out rows. The anomaly score is LEIE-independent, so its
  two rows should differ only by sampling noise.
- Held-out hits lean on name+state matching (noisier) and the 116 val NPIs
  (used for early stopping — mildly seen). Ground truth is caught fraud.
- Same caveats as the original backtest: precision/lift not recall, in-sample
  feature window, ~10% of LEIE rows carry an NPI.
""")
    log(f"    → {OUT_DIR}")
    for k in ("h2h_all", "h2h_held_out"):
        log(f"    {k}: model {results[k]['model']['top_decile_lift']}x "
            f"vs anomaly {results[k]['anomaly']['top_decile_lift']}x")


if __name__ == "__main__":
    main()

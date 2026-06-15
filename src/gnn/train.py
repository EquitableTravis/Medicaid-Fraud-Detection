"""
train.py (gnn) — Phase 7: train FraudSAGE (GATE 6 smoke + GATE 7 training).

FULL-BATCH training: the 617k-node graph fits in memory, so each epoch is one
forward/backward over the whole graph with the focal loss masked to the labeled
TRAIN nodes (unlabeled + val/test nodes still pass messages — inductive — but get
no gradient). This is exact (no neighbor-sampling variance) and avoids the
pyg-lib/torch-sparse sampler backend, which has no wheels for torch 2.12. The
model is unchanged inductive GraphSAGE; mini-batch NeighborLoader can be swapped
back in for a much larger graph later.

**Early-stop on val PR-AUC, never ROC-AUC** (ROC misleadingly high at 0.13%
prevalence). ≥3 seeds — a single run is not a result; report the PR-AUC range.

Artifacts (~/Desktop/Data/Model/gnn/): model_seed{S}.pt per seed, train_report.json.

Run:
    python -m src.gnn.train --smoke      # GATE 6 only (forward + shape check)
    python -m src.gnn.train              # GATE 7 (3 seeds)
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.data import Data

from . import config
from .model import FraudSAGE, focal_loss, pick_device

MAX_EPOCHS, PATIENCE = 200, 20
SEEDS = [0, 1, 2]
HP = dict(hidden=128, dropout=0.3, layers=2, lr=1e-3, wd=5e-4)   # overridden by CLI


def log(m=""):
    print(m, flush=True)


def load_graph(device):
    d = config.GNN_DATA_DIR
    Xn = np.load(d / "X_num.npy")                       # [N,47] f32
    Xc = np.load(d / "X_cat.npy")                       # [N,3] i64
    ei = np.load(d / "edges_all.npy")                   # [2,E] i64
    meta = json.loads((d / "features_meta.json").read_text())
    nodes = pd.read_parquet(config.NODES_PARQUET, columns=["row_index", "label_status"]
                            ).sort_values("row_index")
    sp = pd.read_parquet(d / "split_masks.parquet", columns=["row_index", "split"]
                         ).sort_values("row_index")["split"].to_numpy()
    ls = nodes["label_status"].to_numpy()
    y = (ls == "positive").astype(np.float32)
    labeled = np.isin(ls, ["positive", "reliable_neg"])
    data = Data(x=torch.from_numpy(Xn), edge_index=torch.from_numpy(ei),
                x_cat=torch.from_numpy(Xc), y=torch.from_numpy(y)).to(device)
    masks = {s: torch.from_numpy((sp == s) & labeled).to(device)
             for s in ["train", "val", "test"]}
    return data, masks, [meta["vocab_sizes"][c] for c in meta["categorical_features"]], meta


@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    logits = model(data.x, data.x_cat, data.edge_index)
    s = logits[mask].float().cpu().numpy()
    yt = data.y[mask].cpu().numpy()
    return average_precision_score(yt, s), roc_auc_score(yt, s)


def train_one(seed, data, masks, vocab, tag=""):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FraudSAGE(data.x.shape[1], vocab, hidden=HP["hidden"],
                      dropout=HP["dropout"], layers=HP["layers"]).to(data.x.device)
    opt = torch.optim.Adam(model.parameters(), lr=HP["lr"], weight_decay=HP["wd"])
    best_val, best_state, best_roc, wait = -1.0, None, 0.0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        out = model(data.x, data.x_cat, data.edge_index)
        loss = focal_loss(out[masks["train"]], data.y[masks["train"]])
        loss.backward(); opt.step()
        vpr, vroc = evaluate(model, data, masks["val"])
        if vpr > best_val:
            best_val, best_roc, wait = vpr, vroc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        if ep % 10 == 0 or wait == 0:
            log(f"    seed{seed} ep{ep:03d} loss {float(loss):.4f} | val PR-AUC {vpr:.4f} "
                f"ROC {vroc:.3f}{' *' if wait == 0 else ''}")
        if wait >= PATIENCE:
            log(f"    seed{seed} early stop @ep{ep} (best val PR-AUC {best_val:.4f})")
            break
    model.load_state_dict(best_state)
    tpr, troc = evaluate(model, data, masks["test"])
    torch.save(best_state, config.GNN_DATA_DIR / f"model{tag}_seed{seed}.pt")
    return {"seed": seed, "val_pr_auc": best_val, "val_roc_auc": best_roc,
            "test_pr_auc": tpr, "test_roc_auc": troc}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="GATE 6 only")
    ap.add_argument("--device", default="cpu", help="cpu (default, full-batch safe) or mps")
    ap.add_argument("--tag", default="", help="suffix for artifacts (e.g. _reg) to not clobber")
    for k in HP:
        ap.add_argument(f"--{k}", type=type(HP[k]), default=HP[k])
    args = ap.parse_args()
    HP.update({k: getattr(args, k) for k in HP})
    device = pick_device("mps") if args.device == "mps" else "cpu"
    log(f"device: {device}")
    data, masks, vocab, meta = load_graph(device)
    log(f"graph: {data.num_nodes:,} nodes, {data.edge_index.shape[1]:,} edges | "
        f"vocab {vocab} | numeric {data.x.shape[1]}")
    for s in ["train", "val", "test"]:
        log(f"  {s} labeled: {int(masks[s].sum()):,} "
            f"(positives {int(data.y[masks[s]].sum())})")

    if args.smoke:                                       # GATE 6
        model = FraudSAGE(data.x.shape[1], vocab).to(device)
        out = model(data.x, data.x_cat, data.edge_index)
        seed_logits = out[masks["train"]]
        ntr = int(masks["train"].sum())
        log(f"\nGATE 6: full-graph forward → logits {tuple(out.shape)} "
            f"(expect ({data.num_nodes},)); train-masked {tuple(seed_logits.shape)} "
            f"(expect ({ntr},)) → "
            f"{'OK' if out.shape == (data.num_nodes,) and seed_logits.shape == (ntr,) else 'FAIL'}")
        return

    log(f"\nGATE 7: training ≥3 seeds (early-stop on val PR-AUC) | HP {HP} tag '{args.tag}'")
    results = [train_one(s, data, masks, vocab, args.tag) for s in SEEDS]
    vpr = [r["val_pr_auc"] for r in results]
    tpr = [r["test_pr_auc"] for r in results]
    (config.GNN_DATA_DIR / f"train_report{args.tag}.json").write_text(json.dumps(
        {"config": {"full_batch": True, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                    "seeds": SEEDS, "device": device, **HP}, "results": results,
         "val_pr_auc_range": [min(vpr), max(vpr)], "val_pr_auc_mean": float(np.mean(vpr)),
         "test_pr_auc_range": [min(tpr), max(tpr)], "test_pr_auc_mean": float(np.mean(tpr))},
        indent=2))
    log("\n" + "=" * 60 + "\nGATE 7 — ≥3-seed results\n" + "=" * 60)
    for r in results:
        log(f"  seed{r['seed']}: val PR-AUC {r['val_pr_auc']:.4f} (ROC {r['val_roc_auc']:.3f}) "
            f"| test PR-AUC {r['test_pr_auc']:.4f}")
    log(f"  VAL PR-AUC range: {min(vpr):.4f}–{max(vpr):.4f} (mean {np.mean(vpr):.4f})")
    log(f"  TEST PR-AUC range: {min(tpr):.4f}–{max(tpr):.4f} (mean {np.mean(tpr):.4f})")
    log(f"  (LightGBM tabular baseline: held-out PR-AUC 0.465; honest head-to-head on the "
        f"SAME temporal split is Phase 8)")


if __name__ == "__main__":
    main()

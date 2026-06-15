"""
model.py (gnn) — Phase 6: the FraudSAGE architecture + focal loss.

Inductive, homogeneous GraphSAGE:
  numeric features  ─┐
  entity embedding   ├─ concat → 2× SAGEConv(+ReLU+dropout) → linear head → 1 logit/node
  taxonomy embedding │
  state embedding   ─┘
Inductive (mean aggregator) so NPIs unseen at train time can be scored. The 3
categoricals get learnable embeddings sized to their vocab.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


def emb_dim(vocab: int) -> int:
    return int(min(16, max(2, np.ceil(np.sqrt(vocab)))))


class FraudSAGE(nn.Module):
    def __init__(self, num_dim, vocab_sizes, hidden=128, dropout=0.3, layers=2):
        super().__init__()
        self.dropout = dropout
        self.embs = nn.ModuleList([nn.Embedding(v, emb_dim(v)) for v in vocab_sizes])
        in_dim = num_dim + sum(emb_dim(v) for v in vocab_sizes)
        dims = [in_dim] + [hidden] * layers
        self.convs = nn.ModuleList([SAGEConv(dims[i], dims[i + 1]) for i in range(layers)])
        self.head = nn.Linear(hidden, 1)

    def forward(self, x_num, x_cat, edge_index):
        h = torch.cat([x_num] + [e(x_cat[:, i]) for i, e in enumerate(self.embs)], dim=1)
        for conv in self.convs:
            h = F.dropout(F.relu(conv(h, edge_index)), self.dropout, self.training)
        return self.head(h).squeeze(-1)          # logit per node


def focal_loss(logits, targets, alpha=0.9, gamma=2.0):
    """Binary focal loss — down-weights easy negatives, up-weights the rare positive.
    alpha is the weight on the positive class (prior ≈ 0.13% positives)."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    a_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (a_t * (1 - p_t) ** gamma * ce).mean()


def pick_device(prefer="mps"):
    """MPS if it actually runs a SAGEConv (PyG MPS scatter support is patchy), else CPU."""
    if prefer == "mps" and torch.backends.mps.is_available():
        try:
            x = torch.randn(4, 3, device="mps")
            ei = torch.tensor([[0, 1, 2], [1, 2, 3]], device="mps")
            SAGEConv(3, 5).to("mps")(x, ei)
            return "mps"
        except Exception as e:                    # noqa: BLE001
            print(f"  MPS smoke failed ({type(e).__name__}); using CPU", flush=True)
    return "cpu"

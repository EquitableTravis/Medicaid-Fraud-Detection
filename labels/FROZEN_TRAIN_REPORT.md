# Frozen-package training — forward-label results

package: `/Users/traviswaters/Desktop/Data/preclean/trey/frozen_2023-12` | seeds: [42, 43, 44] | 5-fold GroupKFold(group_id), out-of-fold scoring, eval on eligible rows vs future bans

| variant | cols | fwd ROC-AUC | fwd PR-AUC | lift@10% | recall@1000 | P@100 |
|---|---|---|---|---|---|---|
| A_all_trainable | 116 | 0.7177 [0.7147-0.7236] | 0.0066 | 3.61 | 0.025 | 0.043 |
| B_strict_no_current_state | 91 | 0.7133 [0.7064-0.7236] | 0.0060 | 3.38 | 0.023 | 0.033 |

current_state contribution (A − B, fwd ROC-AUC): **+0.0044**

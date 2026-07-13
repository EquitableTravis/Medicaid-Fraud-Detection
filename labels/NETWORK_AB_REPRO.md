# Network A/B — independent reproduction + audit extensions

package: `/Users/traviswaters/Desktop/Data/preclean/trey/frozen_2023-12` | seeds [42, 43, 44] | matched 3:1 on (taxonomy, state, net_paid quartile) ladder | grouped 5-fold CV | cluster bootstrap CIs (1,000)

| label | arm | ROC with | ROC without | ΔROC [95% CI] | ΔPR [95% CI] |
|---|---|---|---|---|---|
| all_forward | full_net | 0.7389 | 0.6900 | +0.0491 [+0.0379,+0.0605] | +0.0657 [+0.0459,+0.0848] |
| all_forward | structural | 0.7404 | 0.6900 | +0.0505 [+0.0394,+0.0619] | +0.0678 [+0.0483,+0.0874] |
| all_forward | rpd_only | 0.6907 | 0.6900 | +0.0008 [-0.0050,+0.0066] | +0.0003 [-0.0106,+0.0112] |
| all_forward | shell_deprox | 0.7346 | 0.6900 | +0.0448 [+0.0332,+0.0563] | +0.0608 [+0.0405,+0.0813] |
| late_only_gt6mo | full_net | 0.7198 | 0.6803 | +0.0398 [+0.0268,+0.0527] | +0.0621 [+0.0384,+0.0851] |
| late_only_gt6mo | structural | 0.7178 | 0.6803 | +0.0377 [+0.0255,+0.0497] | +0.0625 [+0.0409,+0.0844] |
| late_only_gt6mo | rpd_only | 0.6773 | 0.6803 | -0.0030 [-0.0063,+0.0003] | -0.0009 [-0.0065,+0.0045] |
| late_only_gt6mo | shell_deprox | 0.7151 | 0.6803 | +0.0351 [+0.0222,+0.0475] | +0.0586 [+0.0359,+0.0814] |

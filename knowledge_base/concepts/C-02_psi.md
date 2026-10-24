---
id: C-02
title: Population Stability Index (PSI)
category: concept
tags: psi, population stability index, severity, binning, thresholds
---

# C-02 — Population Stability Index (PSI)

PSI measures **how much** a distribution has shifted between a reference sample (training data) and a new sample
(production batch).

Formula: PSI = Σ (actual% − expected%) × ln(actual% / expected%), summed over bins.
- Numeric features: 10 bins from the reference quantiles (so each reference bin holds ~10% of rows).
- Categorical features: one bin per category.
- Empty bins are floored at a tiny value (0.0001) to avoid log(0).

Industry rule of thumb used by this project:
| PSI | Meaning |
|---|---|
| < 0.10 | Stable — no meaningful change |
| 0.10 – 0.25 | Moderate shift — monitor / alert |
| > 0.25 | Major shift — investigate, likely retrain |

PSI is about **effect size**. With large samples a KS test can be significant for tiny shifts; PSI tells whether
the shift is big enough to matter. The Watchdog uses both (see C-03 and the Critic rule in P-03).

---
id: C-08
title: Champion vs challenger evaluation
category: concept
tags: champion, challenger, retrain, auc, holdout, deployment
---

# C-08 — Champion vs challenger evaluation

The **champion** is the model currently in production. When the Watchdog decides to Retrain, the Reporter trains a
**challenger** on reference data plus recent labelled data (recent rows weighted 3×) and evaluates both models on
**fresh, unseen** data (2,000 rows).

The challenger can be proposed only if its AUC beats the champion's by at least 0.01 (P-04), and it is deployed
only after a human approves. This prevents replacing a working model with one that only looks better on the data it
was trained on.

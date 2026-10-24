---
id: P-04
title: Model deployment rule
category: policy
tags: deployment, approval, human in the loop, auc gain, registry
---

# P-04 — Model deployment rule

A retrained challenger replaces the champion only when **both** are true:
1. Its AUC on fresh, unseen data is at least **0.01** higher than the champion's.
2. A human reviewer clicks **Approve & deploy**.

Every deployment is recorded in the model registry (version, time, training data, AUC, approver), giving a full
audit trail. Rejected challengers are archived.

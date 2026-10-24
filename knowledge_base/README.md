# Knowledge Base

Documents the agents retrieve and cite during every Watchdog run.

| ID | Title | Category |
|---|---|---|
| [C-01](concepts/C-01_data_drift.md) | Data drift | concept |
| [C-02](concepts/C-02_psi.md) | Population Stability Index (PSI) | concept |
| [C-03](concepts/C-03_ks_test.md) | Kolmogorov–Smirnov (KS) test | concept |
| [C-04](concepts/C-04_chi_square.md) | Chi-square test of independence | concept |
| [C-05](concepts/C-05_concept_drift.md) | Concept drift vs data drift | concept |
| [C-06](concepts/C-06_prediction_drift.md) | Prediction drift | concept |
| [C-07](concepts/C-07_bootstrap.md) | Bootstrap re-check | concept |
| [C-08](concepts/C-08_champion_challenger.md) | Champion vs challenger evaluation | concept |
| [D-01](domain/D-01_credit_risk.md) | Credit risk models in digital lending | domain |
| [D-02](domain/D-02_label_delay.md) | Label delay | domain |
| [D-03](domain/D-03_audit_explainability.md) | Auditability and explainability | domain |
| [D-04](domain/D-04_gig_workers.md) | Gig-worker applicants | domain |
| [P-01](policy/P-01_decision_policy.md) | Decision policy | policy |
| [P-02](policy/P-02_critical_features.md) | Critical features | policy |
| [P-03](policy/P-03_critic_rules.md) | Critic validation rules | policy |
| [P-04](policy/P-04_deployment_rule.md) | Model deployment rule | policy |
| [RB-01](runbooks/RB-01_no_action.md) | Runbook: No action | runbook |
| [RB-02](runbooks/RB-02_alert.md) | Runbook: Alert (moderate drift) | runbook |
| [RB-03](runbooks/RB-03_retrain.md) | Runbook: Retrain | runbook |
| [RB-04](runbooks/RB-04_fix_pipeline.md) | Runbook: Fix pipeline (data-quality incident) | runbook |
| [RB-05](runbooks/RB-05_escalation.md) | Runbook: Escalation to a human | runbook |

Run history (agent memory) is stored separately in `data/memory.jsonl` and is not committed.

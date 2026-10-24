---
id: P-02
title: Critical features
category: policy
tags: critical features, credit score, income, loan amount, employment type
---

# P-02 — Critical features

Critical features are the ones that drive credit risk the most and whose drift is most likely to harm decisions:

- **credit_score** — primary signal of repayment history.
- **monthly_income** — affordability.
- **loan_amount** — exposure; combined with income gives loan-to-income.
- **employment_type** — income stability (salaried vs self-employed vs gig).

Non-critical: age, city_tier. Major drift on a non-critical feature raises an Alert, not a Retrain.

---
id: D-01
title: Credit risk models in digital lending
category: domain
tags: credit risk, lending, default, approval, nbfc, digital lending
---

# D-01 — Credit risk models in digital lending

A credit-risk model estimates the probability that an applicant will default. Digital lenders use it to approve or
reject loans in seconds and to set limits.

Two kinds of error cost money:
- Approving a risky applicant → loan loss.
- Rejecting a good applicant → lost revenue and customer.

The champion in this project is a Random Forest on age, monthly income, loan amount, credit score, employment type
and city tier. Its value depends on applicants continuing to look like the training population — which is exactly
what drift monitoring protects.

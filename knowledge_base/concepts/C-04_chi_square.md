---
id: C-04
title: Chi-square test of independence
category: concept
tags: chi-square, categorical, contingency table, p-value
---

# C-04 — Chi-square test of independence

For **categorical** features the Watchdog builds a contingency table of category counts (reference vs production)
and runs a Chi-square test. A p-value below 0.05 means the category mix has changed significantly.

Example: employment_type shifting from 10% gig workers to 43% gig workers gives a huge Chi-square statistic
and a p-value close to zero.

Used by the Planner for: employment_type, city_tier (city_tier is stored as 1/2/3 but treated as a category).
PSI is computed on category shares alongside the test.

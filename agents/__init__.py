"""Agents for Model Drift Watchdog.

planner   - builds the task plan and re-plans on bad data      (Planner)
profiler  - checks data quality, widens the window on request  (Research)
detector  - runs KS / Chi-square / PSI and bootstrap re-checks (Executor)
decision  - applies the locked policy                          (Business Automation)
critic    - validates evidence and decision, retries/escalates (Critic)
reporter  - report, alert, incident, champion vs challenger    (Executor + Automation)
"""

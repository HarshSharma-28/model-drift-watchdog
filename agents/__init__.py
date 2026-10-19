"""Agents for Model Drift Watchdog.

planner   - builds and re-plans the task list
profiler  - checks data quality
detector  - runs drift tests
decision  - applies the policy
critic    - validates and triggers retries / escalation
reporter  - writes the report, alert and champion vs challenger comparison
"""

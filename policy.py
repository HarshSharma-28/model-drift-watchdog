"""Locked thresholds and decision rules for Model Drift Watchdog.

Every agent reads its limits from here, so the policy lives in one place
and can be shown to the panel as-is.
"""

# --- Drift thresholds -------------------------------------------------------
PSI_MODERATE = 0.10          # PSI < 0.10 -> stable
PSI_MAJOR = 0.25             # 0.10-0.25 -> moderate, > 0.25 -> major
P_VALUE_ALPHA = 0.05         # KS / Chi-square p < 0.05 -> drifted

# --- Data quality -----------------------------------------------------------
MAX_MISSING_RATE = 0.10      # > 10% missing on any feature -> data-quality issue
MIN_SAMPLE_SIZE = 500        # Critic: smaller batches trigger a wider window

# --- Performance (only when labels are available) ---------------------------
MAX_ACCURACY_DROP = 0.05     # > 5 percentage points -> retrain

# --- Features ---------------------------------------------------------------
NUMERIC_FEATURES = ["age", "monthly_income", "loan_amount", "credit_score"]
CATEGORICAL_FEATURES = ["employment_type", "city_tier"]
TARGET = "default"
CRITICAL_FEATURES = ["credit_score", "monthly_income", "loan_amount", "employment_type"]

# --- Orchestration ----------------------------------------------------------
MAX_RETRIES = 2              # Critic retries before escalating to a human

# --- Deployment -------------------------------------------------------------
MIN_AUC_GAIN = 0.01          # Challenger must beat champion AUC by this much

# --- Possible decisions -----------------------------------------------------
NO_ACTION = "No action"
ALERT = "Alert"
RETRAIN = "Retrain"
FIX_PIPELINE = "Fix pipeline"

"""
=============================================================
RavenStack SaaS — Step 1: EDA, KPIs, and Business Insights
=============================================================
Purpose:
- Load all 5 CSV files
- Build a master customer table
- Calculate business KPIs
- Generate charts for business analysis


=============================================================
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 150, "font.family": "DejaVu Sans"})


# ============================================================
# 1. PATH SETUP
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def require_file(path: Path):
    """Raise a clear error if a file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")


def load_csv(path: Path, parse_dates=None):
    """Load CSV safely with friendly error handling."""
    require_file(path)
    return pd.read_csv(path, parse_dates=parse_dates)


# ============================================================
# 2. LOAD DATA
# ============================================================
print("Loading RavenStack data...")

accounts = load_csv(DATA_DIR / "ravenstack_accounts.csv", parse_dates=["signup_date"])
subs = load_csv(DATA_DIR / "ravenstack_subscriptions.csv", parse_dates=["start_date", "end_date"])
usage = load_csv(DATA_DIR / "ravenstack_feature_usage.csv", parse_dates=["usage_date"])
tickets = load_csv(DATA_DIR / "ravenstack_support_tickets.csv", parse_dates=["submitted_at", "closed_at"])
churn = load_csv(DATA_DIR / "ravenstack_churn_events.csv", parse_dates=["churn_date"])

print(f"accounts        : {accounts.shape}")
print(f"subscriptions   : {subs.shape}")
print(f"feature_usage   : {usage.shape}")
print(f"support_tickets : {tickets.shape}")
print(f"churn_events    : {churn.shape}")


# ============================================================
# 3. BUILD MASTER TABLE
# ============================================================
print("\nBuilding master analysis table...")

# Latest subscription per account
latest_sub_idx = subs.groupby("account_id")["start_date"].idxmax()
latest_sub = subs.loc[latest_sub_idx, [
    "account_id",
    "plan_tier",
    "mrr_amount",
    "arr_amount",
    "billing_frequency",
    "is_trial",
    "upgrade_flag",
    "downgrade_flag",
    "churn_flag",
    "auto_renew_flag"
]].copy()

latest_sub = latest_sub.rename(columns={
    "plan_tier": "sub_plan_tier",
    "is_trial": "sub_is_trial",
    "churn_flag": "sub_churn_flag"
})

# Support summary per account
support_summary = tickets.groupby("account_id").agg(
    ticket_count=("ticket_id", "count"),
    avg_resolution_hrs=("resolution_time_hours", "mean"),
    avg_first_response_mins=("first_response_time_minutes", "mean"),
    avg_satisfaction=("satisfaction_score", "mean"),
    escalation_count=("escalation_flag", "sum"),
    urgent_ticket_count=("priority", lambda x: (x == "urgent").sum())
).reset_index()

# Feature usage summary per account
sub_account = subs[["subscription_id", "account_id"]].drop_duplicates()
usage_joined = usage.merge(sub_account, on="subscription_id", how="left")

usage_summary = usage_joined.groupby("account_id").agg(
    total_usage_events=("usage_id", "count"),
    total_usage_count=("usage_count", "sum"),
    avg_duration_secs=("usage_duration_secs", "mean"),
    total_errors=("error_count", "sum"),
    beta_feature_uses=("is_beta_feature", "sum"),
    unique_features=("feature_name", "nunique")
).reset_index()

# Churn summary per account
churn_summary = churn.groupby("account_id").agg(
    churn_reason=("reason_code", "first"),
    refund_usd=("refund_amount_usd", "sum"),
    is_reactivation=("is_reactivation", "max")
).reset_index()

# Master join
master = (
    accounts
    .merge(latest_sub, on="account_id", how="left")
    .merge(support_summary, on="account_id", how="left")
    .merge(usage_summary, on="account_id", how="left")
    .merge(churn_summary, on="account_id", how="left")
)

# Keep both churn flags if available, but use one consistent column
if "churn_flag" in master.columns and "sub_churn_flag" in master.columns:
    master["churn_flag"] = master["churn_flag"].fillna(master["sub_churn_flag"])

# Fill missing numeric values that mean "no activity"
zero_cols = [
    "ticket_count",
    "escalation_count",
    "urgent_ticket_count",
    "total_usage_events",
    "total_usage_count",
    "total_errors",
    "beta_feature_uses",
]
for col in zero_cols:
    if col in master.columns:
        master[col] = master[col].fillna(0)

# Convert to numeric where needed
num_cols = [
    "mrr_amount", "arr_amount", "avg_resolution_hrs", "avg_first_response_mins",
    "avg_satisfaction", "refund_usd", "avg_duration_secs"
]
for col in num_cols:
    if col in master.columns:
        master[col] = pd.to_numeric(master[col], errors="coerce")

# Tenure in days
master["tenure_days"] = (pd.Timestamp.today().normalize() - master["signup_date"]).dt.days

# Helpful labels
master["churn_label"] = master["churn_flag"].map({True: "churned", False: "retained"})

print(f"Master table shape: {master.shape}")
print(master[["account_id", "account_name", "industry", "mrr_amount", "churn_flag"]].head(3))


# Save master table for later steps
master.to_csv(OUTPUT_DIR / "master_table.csv", index=False)
print(f"Saved: {OUTPUT_DIR / 'master_table.csv'}")


# ============================================================
# 4. BUSINESS KPIs
# ============================================================
print("\n" + "=" * 60)
print("BUSINESS KPIs — RavenStack")
print("=" * 60)

total_accounts = len(master)
churn_rate = master["churn_flag"].mean() * 100
retention_rate = 100 - churn_rate
avg_mrr = master["mrr_amount"].mean()
avg_arr = master["arr_amount"].mean()
revenue_at_risk = master.loc[master["churn_flag"] == True, "mrr_amount"].sum()
avg_resolution = master["avg_resolution_hrs"].mean()
avg_first_resp = master["avg_first_response_mins"].mean()
avg_satisfaction = master["avg_satisfaction"].mean()

trial_users = master[master["sub_is_trial"] == True]
trial_churn_rate = trial_users["churn_flag"].mean() * 100 if len(trial_users) > 0 else np.nan

kpis = {
    "Total Accounts": total_accounts,
    "Churn Rate (%)": f"{churn_rate:.1f}",
    "Retention Rate (%)": f"{retention_rate:.1f}",
    "Average MRR (USD)": f"${avg_mrr:,.0f}",
    "Average ARR (USD)": f"${avg_arr:,.0f}",
    "Revenue at Risk (MRR, USD)": f"${revenue_at_risk:,.0f}",
    "Avg Support Resolution (hrs)": f"{avg_resolution:.1f}",
    "Avg First Response (min)": f"{avg_first_resp:.0f}",
    "Avg Satisfaction Score": f"{avg_satisfaction:.2f} / 5",
    "Trial Churn Rate (%)": f"{trial_churn_rate:.1f}" if not np.isnan(trial_churn_rate) else "N/A",
}

for k, v in kpis.items():
    print(f"{k:<35}: {v}")


# ============================================================
# 5. CHART HELPERS
# ============================================================
def save_chart(filename: str):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# ============================================================
# 6. CHARTS
# ============================================================

# 1) Churn by plan tier
if "sub_plan_tier" in master.columns:
    churn_by_plan = master.groupby("sub_plan_tier")["churn_flag"].mean().mul(100).sort_values()
    plt.figure(figsize=(7, 4))
    bars = plt.bar(churn_by_plan.index.astype(str), churn_by_plan.values, color=["#2ECC71", "#E67E22", "#E74C3C"])
    plt.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    plt.title("Churn Rate by Subscription Plan Tier", fontsize=13, fontweight="bold")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Plan Tier")
    plt.ylim(0, max(churn_by_plan.values) + 10)
    save_chart("01_churn_by_plan_tier.png")

# 2) Churn by industry
if "industry" in master.columns:
    churn_by_ind = master.groupby("industry")["churn_flag"].mean().mul(100).sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    bars = plt.barh(churn_by_ind.index.astype(str), churn_by_ind.values, color="#3498DB")
    plt.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    plt.title("Churn Rate by Industry", fontsize=13, fontweight="bold")
    plt.xlabel("Churn Rate (%)")
    save_chart("02_churn_by_industry.png")

# 3) Churn by referral source
if "referral_source" in master.columns:
    churn_by_ref = master.groupby("referral_source")["churn_flag"].mean().mul(100).sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    bars = plt.barh(churn_by_ref.index.astype(str), churn_by_ref.values, color="#9B59B6")
    plt.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    plt.title("Churn Rate by Referral Source", fontsize=13, fontweight="bold")
    plt.xlabel("Churn Rate (%)")
    save_chart("03_churn_by_referral_source.png")

# 4) Churn reasons
if "reason_code" in churn.columns:
    reason_counts = churn["reason_code"].value_counts()
    plt.figure(figsize=(7, 4))
    bars = plt.bar(reason_counts.index.astype(str), reason_counts.values, color=sns.color_palette("Set2", len(reason_counts)))
    plt.bar_label(bars, padding=4, fontsize=10)
    plt.title("Churn Reasons Distribution", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Churned Accounts")
    plt.xlabel("Reason Code")
    save_chart("04_churn_reasons.png")

# 5) MRR distribution: churned vs retained
if "mrr_amount" in master.columns:
    plt.figure(figsize=(8, 4))
    for label, color in {"retained": "#2ECC71", "churned": "#E74C3C"}.items():
        subset = master.loc[master["churn_label"] == label, "mrr_amount"].dropna()
        if len(subset) > 0:
            plt.hist(subset, bins=25, alpha=0.6, label=label, color=color)
    plt.title("MRR Distribution: Churned vs Retained", fontsize=13, fontweight="bold")
    plt.xlabel("MRR (USD)")
    plt.ylabel("Count")
    plt.legend()
    save_chart("05_mrr_distribution_churn.png")

# 6) Satisfaction vs churn
if "avg_satisfaction" in master.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=master,
        x="churn_label",
        y="avg_satisfaction",
        order=["retained", "churned"],
        palette={"retained": "#2ECC71", "churned": "#E74C3C"}
    )
    plt.title("Average Satisfaction Score: Churned vs Retained", fontsize=13, fontweight="bold")
    plt.ylabel("Avg Satisfaction Score (1–5)")
    plt.xlabel("")
    save_chart("06_satisfaction_vs_churn.png")

# 7) Usage events vs churn
if "total_usage_events" in master.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=master,
        x="churn_label",
        y="total_usage_events",
        order=["retained", "churned"],
        palette={"retained": "#2ECC71", "churned": "#E74C3C"}
    )
    plt.title("Total Usage Events: Churned vs Retained", fontsize=13, fontweight="bold")
    plt.ylabel("Total Usage Events")
    plt.xlabel("")
    save_chart("07_usage_events_vs_churn.png")

# 8) Resolution time vs churn
if "avg_resolution_hrs" in master.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=master,
        x="churn_label",
        y="avg_resolution_hrs",
        order=["retained", "churned"],
        palette={"retained": "#2ECC71", "churned": "#E74C3C"}
    )
    plt.title("Average Resolution Time (hrs): Churned vs Retained", fontsize=13, fontweight="bold")
    plt.ylabel("Avg Resolution Time (hours)")
    plt.xlabel("")
    save_chart("08_resolution_time_vs_churn.png")

# 9) Churn by billing frequency
if "billing_frequency" in master.columns:
    churn_by_billing = master.groupby("billing_frequency")["churn_flag"].mean().mul(100).sort_values()
    plt.figure(figsize=(6, 4))
    bars = plt.bar(churn_by_billing.index.astype(str), churn_by_billing.values, color=["#2ECC71", "#E74C3C"])
    plt.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=10)
    plt.title("Churn Rate by Billing Frequency", fontsize=13, fontweight="bold")
    plt.ylabel("Churn Rate (%)")
    plt.xlabel("Billing Frequency")
    save_chart("09_churn_by_billing_frequency.png")

# 10) Errors vs churn
if "total_errors" in master.columns:
    plt.figure(figsize=(7, 4))
    sns.boxplot(
        data=master,
        x="churn_label",
        y="total_errors",
        order=["retained", "churned"],
        palette={"retained": "#2ECC71", "churned": "#E74C3C"}
    )
    plt.title("Total Feature Errors: Churned vs Retained", fontsize=13, fontweight="bold")
    plt.ylabel("Total Error Count")
    plt.xlabel("")
    save_chart("10_errors_vs_churn.png")

print("\nAll charts saved to outputs/")
print("Next step: build segmentation + ML in 02_segmentation_and_ml.py")
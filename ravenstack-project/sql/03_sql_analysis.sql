-- ============================================================
-- Author: Prateek Prakash
-- Project: SaaS Churn & Retention Analysis (RavenStack)
-- Tools: SQL Server, Python, Machine Learning
-- ============================================================

USE Churn_ml;
GO

-- ============================================================
-- 1. CHURN BY PLAN TIER
-- Business Insight: Which plan has highest churn?
-- ============================================================

SELECT 
    plan_tier,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM dbo.ravenstack_accounts
GROUP BY plan_tier
ORDER BY churn_rate_pct DESC;


-- ============================================================
-- 2. CHURN BY INDUSTRY
-- Business Insight: Which industries are most at risk?
-- ============================================================

SELECT 
    industry,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(100.0 * SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS churn_rate_pct
FROM dbo.ravenstack_accounts
GROUP BY industry
ORDER BY churn_rate_pct DESC;


-- ============================================================
-- 3. REVENUE AT RISK
-- Business Insight: How much MRR is at risk due to churn?
-- ============================================================

SELECT 
    SUM(CASE WHEN a.churn_flag = 1 THEN s.mrr_amount ELSE 0 END) AS churned_mrr,
    SUM(CASE WHEN a.churn_flag = 0 THEN s.mrr_amount ELSE 0 END) AS active_mrr,
    SUM(s.mrr_amount) AS total_mrr
FROM dbo.ravenstack_accounts a
JOIN dbo.ravenstack_subscriptions s
    ON a.account_id = s.account_id;


-- ============================================================
-- 4. SLA BREACH ANALYSIS
-- Business Insight: Is poor support driving churn?
-- ============================================================

SELECT 
    priority,
    COUNT(*) AS total_tickets,
    ROUND(AVG(resolution_time_hours),1) AS avg_resolution_hours,
    SUM(
        CASE 
            WHEN priority = 'urgent' AND resolution_time_hours > 4 THEN 1
            WHEN priority = 'high' AND resolution_time_hours > 8 THEN 1
            WHEN priority = 'medium' AND resolution_time_hours > 24 THEN 1
            WHEN priority = 'low' AND resolution_time_hours > 48 THEN 1
            ELSE 0
        END
    ) AS sla_breaches
FROM dbo.ravenstack_support_tickets
GROUP BY priority;


-- ============================================================
-- 5. FEATURE USAGE & ERROR RATE
-- Business Insight: Which features have high usage but also high errors?
-- ============================================================

SELECT 
    feature_name,
    SUM(usage_count) AS total_usage,
    SUM(error_count) AS total_errors,
    ROUND(100.0 * SUM(error_count) / NULLIF(SUM(usage_count),0), 2) AS error_rate_pct
FROM dbo.ravenstack_feature_usage
GROUP BY feature_name
ORDER BY total_usage DESC;


-- ============================================================
-- 6. CHURN REASONS ANALYSIS
-- Business Insight: Why customers are leaving?
-- ============================================================

SELECT 
    reason_code,
    COUNT(*) AS churn_count,
    ROUND(AVG(refund_amount_usd),2) AS avg_refund_usd
FROM dbo.ravenstack_churn_events
GROUP BY reason_code
ORDER BY churn_count DESC;


-- ============================================================
-- 7. TOP CUSTOMERS BY REVENUE
-- Business Insight: Which customers are most valuable?
-- ============================================================

SELECT TOP 10
    a.account_name,
    a.industry,
    MAX(s.mrr_amount) AS mrr
FROM dbo.ravenstack_accounts a
JOIN dbo.ravenstack_subscriptions s
    ON a.account_id = s.account_id
GROUP BY a.account_name, a.industry
ORDER BY mrr DESC;


-- ============================================================
-- (OPTIONAL BONUS) COHORT ANALYSIS
-- Business Insight: How churn changes over time
-- ============================================================

WITH cohort AS (
    SELECT 
        DATEFROMPARTS(YEAR(signup_date), MONTH(signup_date), 1) AS cohort_month,
        churn_flag
    FROM dbo.ravenstack_accounts
)
SELECT 
    cohort_month,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) AS churned,
    ROUND(100.0 * SUM(CASE WHEN churn_flag = 1 THEN 1 ELSE 0 END) / COUNT(*),1) AS churn_rate_pct
FROM cohort
GROUP BY cohort_month
ORDER BY cohort_month;
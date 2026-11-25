WITH totals AS (
    SELECT
        COALESCE((SELECT SUM(net_gmv_usd) FROM {{ ref('fct_daily_gmv') }}), 0.0) AS fct_total,
        COALESCE((SELECT SUM(lifetime_gmv_usd) FROM {{ ref('dim_user_ltv') }}), 0.0) AS dim_total
)
SELECT
    fct_total,
    dim_total,
    ABS(fct_total - dim_total) AS diff
FROM totals
WHERE ABS(fct_total - dim_total) > 0.01

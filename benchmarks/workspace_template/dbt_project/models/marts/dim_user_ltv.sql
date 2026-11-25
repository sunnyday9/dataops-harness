{{ config(materialized='table') }}

WITH user_orders AS (
    SELECT
        user_id,
        COUNT(*) AS lifetime_orders,
        ROUND(SUM(amount_usd), 2) AS lifetime_gmv_usd,
        MIN(order_date) AS first_order_date,
        MAX(order_date) AS last_order_date
    FROM {{ ref('int_orders_usd') }}
    GROUP BY user_id
)
SELECT
    u.user_id,
    u.country,
    u.signup_date,
    COALESCE(o.lifetime_orders, 0) AS lifetime_orders,
    COALESCE(o.lifetime_gmv_usd, 0.0) AS lifetime_gmv_usd,
    o.first_order_date,
    o.last_order_date
FROM {{ ref('stg_users') }} AS u
LEFT JOIN user_orders AS o
    ON u.user_id = o.user_id

{{
    config(
        materialized='incremental',
        unique_key='order_date',
        incremental_strategy='delete+insert'
    )
}}

SELECT
    order_date,
    COUNT(*) AS order_count,
    ROUND(SUM(amount_usd), 2) AS net_gmv_usd
FROM {{ ref('int_orders_usd') }}
{% if is_incremental() %}
WHERE order_date >= (
    SELECT COALESCE(MAX(order_date), DATE '1970-01-01') FROM {{ this }}
)
{% endif %}
GROUP BY order_date

SELECT
    o.order_id,
    o.user_id,
    o.order_date,
    o.currency,
    o.amount_local,
    r.rate_to_usd,
    {{ to_usd('o.amount_local', 'r.rate_to_usd') }} AS amount_usd,
    o.user_country
FROM {{ ref('stg_orders') }} AS o
INNER JOIN {{ ref('stg_exchange_rates') }} AS r
    ON o.order_date = r.rate_date
   AND o.currency = r.currency
WHERE o.status != 'cancelled'

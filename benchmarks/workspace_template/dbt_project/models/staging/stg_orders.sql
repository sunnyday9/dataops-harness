SELECT
    CAST(order_id AS VARCHAR) AS order_id,
    CAST(user_id AS VARCHAR) AS user_id,
    CAST(order_date AS DATE) AS order_date,
    UPPER(TRIM(CAST(currency AS VARCHAR))) AS currency,
    CAST(amount_local AS DOUBLE) AS amount_local,
    LOWER(TRIM(CAST(status AS VARCHAR))) AS status,
    CAST(user_country AS VARCHAR) AS user_country
FROM {{ silver_table('silver_orders') }}

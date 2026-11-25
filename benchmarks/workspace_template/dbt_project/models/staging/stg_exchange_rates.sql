SELECT
    CAST(rate_date AS DATE) AS rate_date,
    UPPER(TRIM(CAST(currency AS VARCHAR))) AS currency,
    CAST(rate_to_usd AS DOUBLE) AS rate_to_usd
FROM {{ silver_table('silver_exchange_rates') }}

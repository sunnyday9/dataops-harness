SELECT
    CAST(user_id AS VARCHAR) AS user_id,
    CAST(signup_date AS DATE) AS signup_date,
    UPPER(TRIM(CAST(country AS VARCHAR))) AS country,
    CAST(email AS VARCHAR) AS email,
    CAST(phone AS VARCHAR) AS phone,
    CAST(national_id AS VARCHAR) AS national_id
FROM {{ silver_table('silver_users') }}

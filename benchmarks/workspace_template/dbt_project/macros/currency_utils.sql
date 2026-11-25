{% macro to_usd(amount_col, rate_col) -%}
ROUND(CAST({{ amount_col }} AS DOUBLE) * CAST({{ rate_col }} AS DOUBLE), 2)
{%- endmacro %}

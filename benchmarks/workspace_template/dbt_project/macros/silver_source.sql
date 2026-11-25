{% macro silver_table(table_name) -%}
delta_scan('{{ env_var("DATAOPS_SILVER_" ~ table_name | upper ~ "_PATH", env_var("DATAOPS_PROD_SILVER_DIR", "../lakehouse/prod/silver") ~ "/" ~ table_name) }}')
{%- endmacro %}

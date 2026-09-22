# Devin-databricks-poc

This repository is a synthetic data engineering project used to test automated
Databricks failure investigation and remediation.

## Scenario

The `daily_orders_pipeline` job contains an intentional schema-related bug.

The input DataFrame contains:

- order_id
- qty
- price

The transformation incorrectly references:

- quantity
- unit_price

Expected behavior:

revenue = qty * price

The purpose of this repository is to test whether an AI coding agent can:

1. Investigate a pipeline failure
2. Identify the root cause
3. Modify the appropriate code
4. Add or update tests
5. Run validation
6. Create a pull request

This repository contains no production code, credentials, or production data.

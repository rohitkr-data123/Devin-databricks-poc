import pytest

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pipeline import (
    Pipeline,
    DataQualityError,
    SchemaValidationError,
)


@pytest.fixture(scope="session")
def spark():

    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("pipeline-tests")
        .config(
            "spark.sql.shuffle.partitions",
            "4",
        )
        .config(
            "spark.sql.adaptive.enabled",
            "true",
        )
        .getOrCreate()
    )

    yield spark

    spark.stop()


@pytest.fixture
def pipeline(spark, tmp_path):

    return Pipeline(
        spark=spark,
        target_path=str(tmp_path / "delta"),
    )


# ---------------------------------------------------------
# Source data helpers
# ---------------------------------------------------------

def orders_df(spark):

    return spark.createDataFrame(
        [
            ("c1", "o1", 100.0, "2026-09-20 10:00:00"),
            ("c2", "o2", 200.0, "2026-09-20 11:00:00"),
            ("c3", "o3", 50.0, "2026-09-21 12:00:00"),
        ],
        [
            "customer_id",
            "order_id",
            "order_amount",
            "order_ts",
        ],
    )


def customers_df(spark):

    return spark.createDataFrame(
        [
            ("c1", "US", "premium"),
            ("c2", "IN", "standard"),
            ("c3", "UK", "premium"),
        ],
        [
            "customer_id",
            "country",
            "segment",
        ],
    )


# ---------------------------------------------------------
# Happy path
# ---------------------------------------------------------

def test_pipeline_happy_path(
    spark,
    pipeline,
):

    orders = orders_df(spark)
    customers = customers_df(spark)

    result = pipeline.run(
        orders,
        customers,
    )

    assert result.count() == 3

    assert {
        "order_date",
        "country",
        "segment",
        "order_count",
        "revenue",
        "avg_order_amount",
    }.issubset(result.columns)


# ---------------------------------------------------------
# Schema failures
# ---------------------------------------------------------

def test_missing_required_column(
    spark,
    pipeline,
):

    df = spark.createDataFrame(
        [
            ("c1", "o1", 100.0),
        ],
        [
            "customer_id",
            "order_id",
            "order_amount",
        ],
    )

    with pytest.raises(SchemaValidationError):
        pipeline.clean_orders(df)


def test_wrong_customer_id_type(
    spark,
    pipeline,
):

    df = spark.createDataFrame(
        [
            (123, "o1", 100.0, "2026-09-20 10:00:00"),
        ],
        [
            "customer_id",
            "order_id",
            "order_amount",
            "order_ts",
        ],
    )

    with pytest.raises(SchemaValidationError):
        pipeline.clean_orders(df)


# ---------------------------------------------------------
# Data quality
# ---------------------------------------------------------

def test_negative_order_amount(
    spark,
    pipeline,
):

    df = spark.createDataFrame(
        [
            ("c1", "o1", -100.0, "2026-09-20 10:00:00"),
        ],
        [
            "customer_id",
            "order_id",
            "order_amount",
            "order_ts",
        ],
    )

    with pytest.raises(DataQualityError):
        pipeline.clean_orders(df)


def test_null_customer_id(
    spark,
    pipeline,
):

    df = spark.createDataFrame(
        [
            (None, "o1", 100.0, "2026-09-20 10:00:00"),
        ],
        [
            "customer_id",
            "order_id",
            "order_amount",
            "order_ts",
        ],
    )

    result = pipeline.clean_orders(df)

    assert result.count() == 0


# ---------------------------------------------------------
# Duplicate records
# ---------------------------------------------------------

def test_duplicate_orders_are_removed(
    spark,
    pipeline,
):

    df = spark.createDataFrame(
        [
            ("c1", "o1", 100.0, "2026-09-20 10:00:00"),
            ("c1", "o1", 150.0, "2026-09-20 12:00:00"),
            ("c2", "o2", 200.0, "2026-09-20 11:00:00"),
        ],
        [
            "customer_id",
            "order_id",
            "order_amount",
            "order_ts",
        ],
    )

    cleaned = pipeline.clean_orders(df)

    result = pipeline.deduplicate_orders(cleaned)

    assert result.count() == 2

    latest = (
        result
        .filter(F.col("order_id") == "o1")
        .first()
    )

    assert latest.order_amount == 150.0


# ---------------------------------------------------------
# Dimension failures
# ---------------------------------------------------------

def test_customer_dimension_missing_columns(
    spark,
    pipeline,
):

    orders = orders_df(spark)

    customers = spark.createDataFrame(
        [
            ("c1",),
        ],
        ["customer_id"],
    )

    with pytest.raises(SchemaValidationError):
        pipeline.enrich_customers(
            orders,
            customers,
        )


def test_orphan_customer_threshold(
    spark,
    pipeline,
):

    orders = spark.createDataFrame(
        [
            ("missing1", "o1", 100.0, "2026-09-20 10:00:00"),
            ("missing2", "o2", 200.0, "2026-09-20 11:00:00"),
            ("c1", "o3", 300.0, "2026-09-20 12:00:00"),
        ],
        [
            "customer_id",
            "order_id",
            "order_amount",
            "order_ts",
        ],
    )

    customers = customers_df(spark)

    enriched = pipeline.enrich_customers(
        orders,
        customers,
    )

    with pytest.raises(DataQualityError):
        pipeline.validate_enriched_data(
            enriched
        )


# ---------------------------------------------------------
# Empty pipeline
# ---------------------------------------------------------

def test_empty_pipeline_fails_quality_check(
    spark,
    pipeline,
):

    df = spark.createDataFrame(
        [],
        """
        customer_id string,
        order_id string,
        order_amount double,
        order_ts string
        """,
    )

    cleaned = pipeline.clean_orders(df)

    customers = customers_df(spark)

    enriched = pipeline.enrich_customers(
        cleaned,
        customers,
    )

    with pytest.raises(DataQualityError):
        pipeline.validate_enriched_data(
            enriched
        )


# ---------------------------------------------------------
# Bad timestamp
# ---------------------------------------------------------

def test_invalid_timestamp_detected(
    spark,
    pipeline,
):

    df = spark.createDataFrame(
        [
            (
                "c1",
                "o1",
                100.0,
                "NOT-A-TIMESTAMP",
            ),
        ],
        [
            "customer_id",
            "order_id",
            "order_amount",
            "order_ts",
        ],
    )

    cleaned = pipeline.clean_orders(df)

    customers = customers_df(spark)

    enriched = pipeline.enrich_customers(
        cleaned,
        customers,
    )

    with pytest.raises(DataQualityError):
        pipeline.validate_enriched_data(
            enriched
        )


# ---------------------------------------------------------
# Aggregate validation
# ---------------------------------------------------------

def test_invalid_aggregate_is_rejected(
    spark,
    pipeline,
):

    df = spark.createDataFrame(
        [
            (
                None,
                "US",
                "premium",
                0,
                -100.0,
                -10.0,
            )
        ],
        [
            "order_date",
            "country",
            "segment",
            "order_count",
            "revenue",
            "avg_order_amount",
        ],
    )

    with pytest.raises(DataQualityError):
        pipeline.validate_aggregates(df)


# ---------------------------------------------------------
# Schema evolution
# ---------------------------------------------------------

def test_extra_columns_do_not_break_pipeline(
    spark,
    pipeline,
):

    df = (
        orders_df(spark)
        .withColumn(
            "new_column",
            F.lit("future-schema")
        )
    )

    result = pipeline.clean_orders(df)

    assert "new_column" in result.columns


# ---------------------------------------------------------
# Join semantics
# ---------------------------------------------------------

def test_customer_enrichment(
    spark,
    pipeline,
):

    orders = orders_df(spark)
    customers = customers_df(spark)

    result = pipeline.enrich_customers(
        orders,
        customers,
    )

    row = (
        result
        .filter(F.col("customer_id") == "c1")
        .first()
    )

    assert row.country == "US"
    assert row.segment == "premium"

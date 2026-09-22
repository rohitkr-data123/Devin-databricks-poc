from pyspark.sql import SparkSession

from src.pipeline import calculate_revenue


def test_calculate_revenue():
    spark = (
        SparkSession.builder
        .master("local[2]")
        .appName("devin-poc-test")
        .getOrCreate()
    )

    data = [
        (1, 2, 10.0),
        (2, 3, 5.0),
    ]

    df = spark.createDataFrame(
        data,
        ["order_id", "qty", "price"]
    )

    result = calculate_revenue(df)

    rows = result.collect()

    assert rows[0]["revenue"] == 20.0
    assert rows[1]["revenue"] == 15.0

    spark.stop()

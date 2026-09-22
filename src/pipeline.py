from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def calculate_revenue(df):

    return df.withColumn(
        "revenue",
        col("quantity") * col("unit_price")
    )

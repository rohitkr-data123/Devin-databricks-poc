from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType,
    TimestampType,
)
from delta.tables import DeltaTable


class PipelineError(Exception):
    pass


class DataQualityError(PipelineError):
    pass


class SchemaValidationError(PipelineError):
    pass


class Pipeline:

    REQUIRED_COLUMNS = {
        "customer_id",
        "order_id",
        "order_amount",
        "order_ts",
    }

    def __init__(self, spark: SparkSession, target_path: str):
        self.spark = spark
        self.target_path = target_path

    # ---------------------------------------------------------
    # 1. Validate incoming schema
    # ---------------------------------------------------------

    def validate_schema(self, df: DataFrame) -> None:
        actual = set(df.columns)
        missing = self.REQUIRED_COLUMNS - actual

        if missing:
            raise SchemaValidationError(
                f"Missing required columns: {sorted(missing)}"
            )

        if df.schema["customer_id"].dataType != StringType():
            raise SchemaValidationError(
                "customer_id must be STRING"
            )

    # ---------------------------------------------------------
    # 2. Clean source data
    # ---------------------------------------------------------

    def clean_orders(self, df: DataFrame) -> DataFrame:

        self.validate_schema(df)

        cleaned = (
            df
            .withColumn(
                "order_amount",
                F.col("order_amount").cast("double")
            )
            .withColumn(
                "order_ts",
                F.to_timestamp("order_ts")
            )
            .filter(F.col("customer_id").isNotNull())
            .filter(F.col("order_id").isNotNull())
            .filter(F.col("order_amount").isNotNull())
        )

        # Negative orders are invalid.
        invalid = cleaned.filter(F.col("order_amount") < 0).count()

        if invalid > 0:
            raise DataQualityError(
                f"Found {invalid} orders with negative amount"
            )

        return cleaned

    # ---------------------------------------------------------
    # 3. Deduplicate
    # ---------------------------------------------------------

    def deduplicate_orders(self, df: DataFrame) -> DataFrame:

        return (
            df
            .withColumn(
                "_rn",
                F.row_number().over(
                    __import__(
                        "pyspark.sql.window",
                        fromlist=["Window"]
                    ).Window
                    .partitionBy("order_id")
                    .orderBy(F.col("order_ts").desc())
                )
            )
            .filter(F.col("_rn") == 1)
            .drop("_rn")
        )

    # ---------------------------------------------------------
    # 4. Join customer dimension
    # ---------------------------------------------------------

    def enrich_customers(
        self,
        orders: DataFrame,
        customers: DataFrame,
    ) -> DataFrame:

        required = {"customer_id", "country", "segment"}

        missing = required - set(customers.columns)

        if missing:
            raise SchemaValidationError(
                f"Customer dimension missing columns: {sorted(missing)}"
            )

        # Broadcast intentionally used here.
        # Devin should inspect whether this is safe for a large dimension.
        return (
            orders.alias("o")
            .join(
                F.broadcast(customers.alias("c")),
                F.col("o.customer_id") == F.col("c.customer_id"),
                "left",
            )
            .select(
                F.col("o.order_id"),
                F.col("o.customer_id"),
                F.col("o.order_amount"),
                F.col("o.order_ts"),
                F.col("c.country"),
                F.col("c.segment"),
            )
        )

    # ---------------------------------------------------------
    # 5. Data quality checks
    # ---------------------------------------------------------

    def validate_enriched_data(self, df: DataFrame) -> None:

        total = df.count()

        if total == 0:
            raise DataQualityError(
                "Pipeline produced zero rows"
            )

        orphan_customers = (
            df.filter(F.col("country").isNull())
            .select("customer_id")
            .distinct()
            .count()
        )

        orphan_ratio = orphan_customers / total

        if orphan_ratio > 0.10:
            raise DataQualityError(
                f"Too many unmatched customers: "
                f"{orphan_ratio:.2%}"
            )

        invalid_amounts = (
            df.filter(
                (F.col("order_amount") <= 0)
                | F.col("order_amount").isNull()
            )
            .count()
        )

        if invalid_amounts > 0:
            raise DataQualityError(
                f"Found {invalid_amounts} invalid order amounts"
            )

    # ---------------------------------------------------------
    # 6. Aggregate
    # ---------------------------------------------------------

    def aggregate(self, df: DataFrame) -> DataFrame:

        return (
            df
            .withColumn(
                "order_date",
                F.to_date("order_ts")
            )
            .groupBy(
                "order_date",
                "country",
                "segment",
            )
            .agg(
                F.countDistinct("order_id").alias("order_count"),
                F.sum("order_amount").alias("revenue"),
                F.avg("order_amount").alias("avg_order_amount"),
            )
        )

    # ---------------------------------------------------------
    # 7. Final validation
    # ---------------------------------------------------------

    def validate_aggregates(self, df: DataFrame) -> None:

        invalid = (
            df.filter(
                (F.col("order_count") <= 0)
                | (F.col("revenue") < 0)
                | F.col("order_date").isNull()
            )
            .count()
        )

        if invalid:
            raise DataQualityError(
                f"{invalid} invalid aggregate rows"
            )

    # ---------------------------------------------------------
    # 8. Delta write
    # ---------------------------------------------------------

    def write_delta(self, df: DataFrame) -> None:

        (
            df
            .write
            .format("delta")
            .mode("append")
            .partitionBy("order_date")
            .save(self.target_path)
        )

    # ---------------------------------------------------------
    # Complete pipeline
    # ---------------------------------------------------------

    def run(
        self,
        orders: DataFrame,
        customers: DataFrame,
    ) -> DataFrame:

        orders = self.clean_orders(orders)

        orders = self.deduplicate_orders(orders)

        enriched = self.enrich_customers(
            orders,
            customers,
        )

        self.validate_enriched_data(enriched)

        result = self.aggregate(enriched)

        self.validate_aggregates(result)

        self.write_delta(result)

        return result

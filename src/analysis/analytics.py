from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date

def show_top_returns_spark(parquet_path, n):
    spark = SparkSession.builder.appName("B3 Top Returns").getOrCreate()
    df = spark.read.parquet(parquet_path)
    df = df.withColumn("data", to_date(col("data"), "yyyyMMdd"))

    df.createOrReplaceTempView("b3")
    query = f"""
        SELECT
            ticker,
            first(preco_fechamento) as preco_inicio,
            last(preco_fechamento) as preco_fim,
            ROUND((last(preco_fechamento) - first(preco_fechamento)) / first(preco_fechamento) * 100, 2) as rendimento_percent
        FROM b3
        GROUP BY ticker
        ORDER BY rendimento_percent DESC
        LIMIT {n}
    """
    result = spark.sql(query)
    result.show(truncate=False)
    spark.stop()

def show_bottom_returns_spark(parquet_path, n):
    spark = SparkSession.builder.appName("B3 Bottom Returns").getOrCreate()
    df = spark.read.parquet(parquet_path)
    df = df.withColumn("data", to_date(col("data"), "yyyyMMdd"))

    df.createOrReplaceTempView("b3")
    query = f"""
        SELECT
            ticker,
            first(preco_fechamento) as preco_inicio,
            last(preco_fechamento) as preco_fim,
            ROUND((last(preco_fechamento) - first(preco_fechamento)) / first(preco_fechamento) * 100, 2) as rendimento_percent
        FROM b3
        GROUP BY ticker
        ORDER BY rendimento_percent ASC
        LIMIT {n}
    """
    result = spark.sql(query)
    result.show(truncate=False)
    spark.stop()

def join_and_analyze_spark(parquet_2024, parquet_2025, n, ordem="desc"):
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, to_date, max as spark_max, min as spark_min

    spark = SparkSession.builder.appName("B3 Join Analysis").getOrCreate()

    df_2024 = spark.read.parquet(parquet_2024)
    df_2025 = spark.read.parquet(parquet_2025)

    df_2024 = df_2024.withColumn("data", to_date(col("data"), "yyyyMMdd"))
    df_2025 = df_2025.withColumn("data", to_date(col("data"), "yyyyMMdd"))

    last_2024 = df_2024.groupBy("ticker").agg(
        spark_max("data").alias("data_2024"),
        spark_max("preco_fechamento").alias("preco_2024")
    )

    first_2025 = df_2025.groupBy("ticker").agg(
        spark_min("data").alias("data_2025"),
        spark_min("preco_fechamento").alias("preco_2025")
    )

    joined = last_2024.join(first_2025, on="ticker", how="inner")
    joined = joined.withColumn(
        "rendimento_percent",
        ((col("preco_2025") - col("preco_2024")) / col("preco_2024") * 100)
    )

    if ordem == "desc":
        joined = joined.orderBy(col("rendimento_percent").desc())
    else:
        joined = joined.orderBy(col("rendimento_percent").asc())

    joined.limit(n).show(truncate=False)
    spark.stop()
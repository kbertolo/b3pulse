from pyspark.sql import SparkSession

def initialize_spark_session(app_name="B3PulseETL"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def spark_parse_b3_file(spark, file_path: str):
    df = spark.read.option("header", "true").csv(file_path)
    df = df.selectExpr(
        "to_date(data, 'yyyyMMdd') as data",
        "bdi",
        "ticker",
        "(abertura / 100) as abertura",
        "(maxima / 100) as maxima",
        "(minima / 100) as minima",
        "(preco_fechamento / 100) as preco_fechamento",
        "cast(volume as float) as volume"
    )
    return df

def save_to_parquet_spark(df, output_path):
    df.write.parquet(output_path, mode="overwrite")
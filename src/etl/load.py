def save_to_parquet(df, output_path):
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].dt.strftime("%Y%m%d")
    df.to_parquet(output_path, index=False)
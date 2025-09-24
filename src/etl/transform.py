import pandas as pd

def parse_b3_file(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="latin1") as f:
        lines = f.readlines()

    lines = [line for line in lines if line.startswith("01")]

    colspecs = [
        (2, 10),    # data
        (12, 24),   # código BDI
        (24, 36),   # código de negociação
        (56, 69),   # preço de abertura
        (69, 82),   # preço máximo
        (82, 95),   # preço mínimo
        (108, 121), # preço de fechamento
        (152, 170), # volume
    ]
    colnames = ["data", "bdi", "ticker", "abertura", "maxima", "minima", "preco_fechamento", "volume"]

    df = pd.read_fwf(pd.io.common.StringIO("".join(lines)), colspecs=colspecs, names=colnames)

    df["data"] = pd.to_datetime(df["data"], format="%Y%m%d")
    df["abertura"] = df["abertura"] / 100
    df["maxima"] = df["maxima"] / 100
    df["minima"] = df["minima"] / 100
    df["preco_fechamento"] = df["preco_fechamento"] / 100
    df["volume"] = df["volume"].astype(float)
    df["ticker"] = df["ticker"].str.strip()

    return df
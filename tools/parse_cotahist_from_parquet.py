# parse_cotahist_from_parquet.py
# Uso:
# python parse_cotahist_from_parquet.py data/parquet_cotas/COTAHIST_A2025.parquet out.csv

import sys
from pathlib import Path
import pandas as pd
import numpy as np

def parse_cotahist_from_series(s: pd.Series) -> pd.DataFrame:
    """
    Recebe uma Series com linhas fixed-width (cada string 245 chars) e retorna DF com colunas decodificadas.
    Usa posições do layout oficial COTAHIST (1-based inclusive).
    """
    # get raw text col as string
    raw = s.astype(str).fillna("")
    # helper: slice using 0-based python indices with end exclusive
    def sl(start, end):
        return raw.str.slice(start-1, end).str.strip()

    df = pd.DataFrame({
        "TIPREG": sl(1,2),
        "DATA": sl(3,10),          # AAAAMMDD
        "CODBDI": sl(11,12),
        "CODNEG": sl(13,24),
        "TPMERC": sl(25,27),
        "NOMRES": sl(28,39),
        "ESPECI": sl(40,49),
        "PRAZOT": sl(50,52),
        "MODREF": sl(53,56),
        "PREABE": sl(57,69),
        "PREMAX": sl(70,82),
        "PREMIN": sl(83,95),
        "PREMED": sl(96,108),
        "PREULT": sl(109,121),
        "PREOFC": sl(122,134),
        "PREOFV": sl(135,147),
        "TOTNEG": sl(148,152),
        "QUATOT": sl(153,170),
        "VOLTOT": sl(171,188),
        "PREEXE": sl(189,201),
        "INDOPC": sl(202,202),
        "DATVEN": sl(203,210),
        "FATCOT": sl(211,217),
        "PTOEXE": sl(218,230),
        "CODISI": sl(231,242),
        "DISMES": sl(243,245)
    })

    # keep only record type '01' (cotações)
    df = df[df["TIPREG"] == "01"].copy()

    # convert date
    df["DATA"] = pd.to_datetime(df["DATA"], format="%Y%m%d", errors="coerce")

    # numeric conversion helper: numeric strings may be empty; interpret as NaN
    def to_int(col):
        return pd.to_numeric(df[col].str.replace(r'\D', '', regex=True), errors="coerce")

    # Fields with V99 or V06 (implied 2 decimals) -> divide by 100
    price_fields = ["PREABE","PREMAX","PREMIN","PREMED","PREULT","PREOFC","PREOFV","PREEXE","PTOEXE","VOLTOT"]
    for f in price_fields:
        # remove non digits, convert, divide by 100
        df[f] = to_int(f) / 100.0

    # integer fields
    int_fields = ["TOTNEG","QUATOT","DISMES","FATCOT"]
    for f in int_fields:
        df[f] = to_int(f).astype("Int64")

    # strip CODNEG (codigo de negociação), already trimmed
    df["CODNEG"] = df["CODNEG"].str.replace(r'\s+', '', regex=True)

    return df

def compute_returns(df: pd.DataFrame, ticker_col="CODNEG", price_col="PREULT"):
    """
    Calcula rendimento percentual (last - first)/first * 100 por ticker
    usando PREULT ordenado por DATA.
    """
    tmp = df[[ticker_col, "DATA", price_col]].dropna(subset=[ticker_col, "DATA", price_col]).copy()
    tmp = tmp.sort_values([ticker_col, "DATA"])
    first = tmp.groupby(ticker_col)[price_col].first()
    last = tmp.groupby(ticker_col)[price_col].last()
    rets = ((last - first) / first) * 100.0
    out = rets.reset_index().rename(columns={0: "rendimento_percent"})
    out.columns = [ticker_col, "rendimento_percent"]
    out = out.sort_values("rendimento_percent", ascending=False)
    return out

def main(parquet_path, out_csv=None, topn=20):
    p = Path(parquet_path)
    if not p.exists():
        print("Arquivo parquet não encontrado:", parquet_path); return 1

    # lê parquet (sua tabela tem uma única coluna com a linha fixed-width)
    df_par = pd.read_parquet(p)
    # pega a primeira coluna (contendo o texto)
    colname = df_par.columns[0]
    print("Coluna raw encontrada:", colname)
    s = df_par[colname]

    parsed = parse_cotahist_from_series(s)
    print("Registros (tipos):", parsed["TIPREG"].value_counts().to_dict())
    print("Exemplo parsed columns:", parsed.columns.tolist())

    # calcular retornos
    rets = compute_returns(parsed, ticker_col="CODNEG", price_col="PREULT")
    print(f"\nTop {topn} por rendimento (CODNEG, %):")
    print(rets.head(topn).to_string(index=False))

    if out_csv:
        rets.to_csv(out_csv, index=False)
        print("Salvo em:", out_csv)

    # opcional: retornar parsed e rets
    return parsed, rets

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python parse_cotahist_from_parquet.py path/to/parquet [out.csv] [topn]")
        sys.exit(1)
    parquet_path = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) >= 3 else None
    topn = int(sys.argv[3]) if len(sys.argv) >= 4 else 20
    main(parquet_path, out_csv, topn)

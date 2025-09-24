# scripts/clean_and_summarize.py
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# --- Funções de parsing (mesma lógica do parser que você rodou) ---
def parse_cotahist_from_series(s: pd.Series) -> pd.DataFrame:
    raw = s.astype(str).fillna("")
    def sl(start, end): return raw.str.slice(start-1, end).str.strip()
    df = pd.DataFrame({
        "TIPREG": sl(1,2),
        "DATA": sl(3,10),
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
    df = df[df["TIPREG"] == "01"].copy()
    df["DATA"] = pd.to_datetime(df["DATA"], format="%Y%m%d", errors="coerce")
    def to_int(col):
        return pd.to_numeric(df[col].str.replace(r'\D', '', regex=True), errors="coerce")
    price_fields = ["PREABE","PREMAX","PREMIN","PREMED","PREULT","PREOFC","PREOFV","PREEXE","PTOEXE","VOLTOT"]
    for f in price_fields:
        df[f] = to_int(f) / 100.0
    int_fields = ["TOTNEG","QUATOT","DISMES","FATCOT"]
    for f in int_fields:
        df[f] = to_int(f).astype("Int64")
    df["CODNEG"] = df["CODNEG"].str.replace(r'\s+', '', regex=True)
    return df

# --- main ---
def main(parquet_path="data/parquet_cotas/COTAHIST_A2025.parquet", topn=10, min_days=10, min_total_volume=1):
    p = Path(parquet_path)
    if not p.exists():
        print("Parquet não encontrado:", p); sys.exit(1)
    df_par = pd.read_parquet(p)
    col = df_par.columns[0]
    s = df_par[col]
    parsed = parse_cotahist_from_series(s)

    # stats por ticker
    tmp = parsed.dropna(subset=["CODNEG","DATA","PREULT"]).copy()
    tmp = tmp.sort_values(["CODNEG","DATA"])
    grouped = tmp.groupby("CODNEG").agg(
        first_price = ("PREULT", "first"),
        last_price  = ("PREULT", "last"),
        days_count   = ("DATA", "nunique"),
        total_trades = ("TOTNEG", "sum"),
        total_volume = ("QUATOT", "sum")
    ).reset_index()
    # calcula rendimento
    grouped["rendimento_percent"] = (grouped["last_price"] - grouped["first_price"]) / grouped["first_price"] * 100.0

    # filtros sugeridos
    filt = (grouped["first_price"] > 0) & (grouped["last_price"] > 0) & \
           (grouped["days_count"] >= min_days) & (grouped["total_trades"] > 0) & (grouped["total_volume"] >= min_total_volume)

    cleaned = grouped[filt].copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna(subset=["rendimento_percent"])

    cleaned = cleaned.sort_values("rendimento_percent", ascending=False).head(topn)
    print("Top after cleaning (sample):")
    print(cleaned.to_string(index=False))

    # build prompt
    lines = [f"{row['CODNEG']}: {row['rendimento_percent']:.2f}%" for _, row in cleaned.iterrows()]
    prompt = "Top {} ativos por rendimento (após filtros: min_days={}, min_volume={}):\n\n{}\n".format(
        len(cleaned), min_days, min_total_volume, "\n".join(lines)
    )
    print("\nPrompt a ser enviado à OpenAI:\n", prompt)

    # try to call openai helper
    try:
        from openai_integration import analyze_text_with_openai
    except Exception as e:
        analyze_text_with_openai = None
        print("OpenAI helper não disponível:", e)

    if analyze_text_with_openai:
        try:
            summary = analyze_text_with_openai("Em Português, gere 3 insights curtos e acionáveis sobre a lista abaixo:\n\n" + prompt, max_tokens=200)
            print("\n--- AI Summary ---\n", summary)
        except Exception as e:
            print("Erro ao chamar OpenAI:", e)
    else:
        print("OpenAI não configurado — copie o prompt e use manualmente.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="?", default="data/parquet_cotas/COTAHIST_A2025.parquet")
    parser.add_argument("--topn", type=int, default=10)
    parser.add_argument("--min_days", type=int, default=10)
    parser.add_argument("--min_volume", type=float, default=1)
    args = parser.parse_args()
    main(args.parquet, args.topn, args.min_days, args.min_volume)

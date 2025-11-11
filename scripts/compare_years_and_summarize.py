"""
scripts/compare_years_and_summarize.py
Comparar duas bases COTAHIST (zip ou parquet) e gerar resumo via OpenAI.

Usage:
  python scripts/compare_years_and_summarize.py path/yearA.ZIP path/yearB.ZIP --topn 20

Outputs (outputs/):
 - compare.csv         : comparação tabela (tickers presentes em qualquer ano)
 - compare_top_delta.csv : top N por delta rendimento
 - compare_prompt.txt  : prompt enviado à OpenAI
 - compare_ai_summary.txt, compare_ai_raw.json (se API chamada)
"""
import os, sys, argparse, json, zipfile
from pathlib import Path
import pandas as pd, numpy as np

# garantir src no path para importar openai helper
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# try to import helper if exists
analyze = None
try:
    from openai_integration import analyze_text_with_openai
    analyze = analyze_text_with_openai
except Exception:
    try:
        from src.openai_integration import analyze_text_with_openai
        analyze = analyze_text_with_openai
    except Exception:
        analyze = None

OUT = Path("outputs")
OUT.mkdir(parents=True, exist_ok=True)

def read_parquet_or_zip(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".parquet":
        dfp = pd.read_parquet(p)
        col = dfp.columns[0]
        s = dfp[col].astype(str).fillna("")
        return s
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as z:
            # find .TXT or .txt file inside (COTAHIST file)
            candidates = [n for n in z.namelist() if n.lower().endswith(".txt")]
            if not candidates:
                raise RuntimeError(f"No .TXT inside zip {path}")
            # prefer first candidate
            name = candidates[0]
            with z.open(name) as fh:
                # read as text; each line is a fixed-width record
                text = fh.read().decode("latin1")  # COTAHIST is latin1/iso-8859-1
                lines = text.splitlines()
                # create Series
                s = pd.Series(lines)
                return s
    # fallback: try reading as text file
    with open(p, "r", encoding="latin1") as fh:
        lines = fh.read().splitlines()
        return pd.Series(lines)

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
        return pd.to_numeric(df[col].str.replace(r'\\D', '', regex=True), errors="coerce")
    price_fields = ["PREABE","PREMAX","PREMIN","PREMED","PREULT","PREOFC","PREOFV","PREEXE","PTOEXE","VOLTOT"]
    for f in price_fields:
        df[f] = to_int(f) / 100.0
    int_fields = ["TOTNEG","QUATOT","DISMES","FATCOT"]
    for f in int_fields:
        df[f] = to_int(f).astype("Int64")
    df["CODNEG"] = df["CODNEG"].str.replace(r'\\s+', '', regex=True)
    return df

def compute_stats(parsed_df, price_col="PREULT"):
    tmp = parsed_df.dropna(subset=["CODNEG","DATA", price_col]).sort_values(["CODNEG","DATA"])
    grouped = tmp.groupby("CODNEG").agg(
        first_price = (price_col, "first"),
        last_price  = (price_col, "last"),
        days_count   = ("DATA", "nunique"),
        total_trades = ("TOTNEG", "sum"),
        total_volume = ("QUATOT", "sum")
    ).reset_index()
    grouped["rendimento_percent"] = (grouped["last_price"] - grouped["first_price"]) / grouped["first_price"] * 100.0
    return grouped

def compare_stats(dfA, dfB, labelA="A", labelB="B"):
    # merge outer
    comp = pd.merge(dfA, dfB, on="CODNEG", how="outer", suffixes=(f"_{labelA}", f"_{labelB}"))
    # indicators for presence
    comp["present_in_A"] = comp["first_price_"+labelA].notna()
    comp["present_in_B"] = comp["first_price_"+labelB].notna()
    # compute deltas where present both
    comp["delta_rendimento"] = comp["rendimento_percent_"+labelB] - comp["rendimento_percent_"+labelA]
    comp["delta_volume"] = comp["total_volume_"+labelB].fillna(0) - comp["total_volume_"+labelA].fillna(0)
    comp["delta_days"] = comp["days_count_"+labelB].fillna(0) - comp["days_count_"+labelA].fillna(0)
    return comp

def build_prompt_for_comparison(comp_df, topn=20, labelA="YearA", labelB="YearB"):
    # top by absolute delta rendimento (both present)
    present_both = comp_df[comp_df["present_in_A"] & comp_df["present_in_B"]].copy()
    present_both = present_both.replace([np.inf, -np.inf], np.nan).dropna(subset=["delta_rendimento"])
    present_both["abs_delta"] = present_both["delta_rendimento"].abs()
    top = present_both.sort_values("abs_delta", ascending=False).head(topn)
    lines = []
    for _, r in top.iterrows():
        lines.append(f"{r['CODNEG']}: {r.get('rendimento_percent_'+labelA, 'NA'):.2f}% -> {r.get('rendimento_percent_'+labelB, 'NA'):.2f}% (delta {r['delta_rendimento']:.2f}%), volA={int(r.get('total_volume_'+labelA,0))}, volB={int(r.get('total_volume_'+labelB,0))}")
    header = (
        f"Compare os resultados entre {labelA} e {labelB}. Abaixo estão os tickers com maior mudança absoluta de rendimento:\n\n"
        "Para cada um dos top listados, responda em Português com:\n"
        "  1) possíveis causas para a mudança (corporate actions, split, mudança de liquidez, erro de dados, novo papel);\n"
        "  2) quais métricas checar imediatamente (dias negociados, volume, TOTNEG, anúncios oficiais);\n"
        "  3) um passo acionável (investigar, excluir, watchlist).\n\n"
    )
    body = "\\n".join(lines)
    prompt = header + body
    return prompt, top

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pathA", help="parquet or zip for year A")
    parser.add_argument("pathB", help="parquet or zip for year B")
    parser.add_argument("--labelA", default="YearA")
    parser.add_argument("--labelB", default="YearB")
    parser.add_argument("--topn", type=int, default=20)
    parser.add_argument("--min_first_price", type=float, default=1.0)
    parser.add_argument("--min_days", type=int, default=10)
    parser.add_argument("--min_volume", type=float, default=100)
    args = parser.parse_args()

    # leitura do arquivo A e B
    print("Lendo", args.pathA)
    sA = read_parquet_or_zip(args.pathA)
    print("Lendo", args.pathB)
    sB = read_parquet_or_zip(args.pathB)

    parsedA = parse_cotahist_from_series(sA)
    parsedB = parse_cotahist_from_series(sB)

    statsA = compute_stats(parsedA)
    statsB = compute_stats(parsedB)

    # apply basic filters (remove penny / low liquidity)
    statsA = statsA[(statsA["first_price"] >= args.min_first_price) & (statsA["days_count"] >= args.min_days) & (statsA["total_volume"] >= args.min_volume)]
    statsB = statsB[(statsB["first_price"] >= args.min_first_price) & (statsB["days_count"] >= args.min_days) & (statsB["total_volume"] >= args.min_volume)]

    statsA = statsA.rename(columns=lambda c: c if c == "CODNEG" else f"{c}_{args.labelA}")
    statsB = statsB.rename(columns=lambda c: c if c == "CODNEG" else f"{c}_{args.labelB}")

    # comparando as estatísticas
    comp = compare_stats(statsA, statsB, labelA=args.labelA, labelB=args.labelB)

    # save comparison CSV
    comp_file = OUT / "compare.csv"
    comp.to_csv(comp_file, index=False)
    print("Saved comparison csv:", comp_file)

    # build prompt using readable labels
    prompt, top_df = build_prompt_for_comparison(comp, topn=args.topn, labelA=args.labelA, labelB=args.labelB)
    (OUT / "compare_prompt.txt").write_text(prompt, encoding="utf-8")
    (OUT / "compare_top_delta.csv").write_text(top_df.to_csv(index=False), encoding="utf-8")

    # call OpenAI if helper available
    if analyze is None:
        print("OpenAI helper não disponível — prompt salvo em outputs/compare_prompt.txt")
        print("Rode com OPENAI_API_KEY e src/openai_integration.py para gerar resumo.")
        return

    print("Chamando OpenAI para analisar diferenças...")
    try:
        # dentro do meu try eu envio para o gpt
        ai_text = analyze(
            prompt, 
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"), 
            max_tokens=int(os.getenv("MAX_TOKENS","300")), 
            save_raw_to=str(OUT / "compare_ai_raw.json"))

        # ja aqui eu recebo o seu retorno em txt 
        (OUT / "compare_ai_summary.txt").write_text(ai_text, encoding="utf-8")
        print("AI summary salvo em:", OUT / "compare_ai_summary.txt")
    except Exception as e:
        print("Erro ao chamar OpenAI:", e)
        print("Prompt salvo:", OUT / "compare_prompt.txt")

if __name__ == "__main__":
    main()

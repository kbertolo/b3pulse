#!/usr/bin/env python3
"""
scripts/run_and_summarize.py
- Parse COTAHIST parquet (fixed-width lines)
- Compute returns, apply filters (env overrides)
- Save outputs/cleaned_top.csv
- Build improved prompt, save outputs/ai_prompt.txt
- If OPENAI_API_KEY present and helper available, call API and save outputs/ai_summary.txt + outputs/ai_raw.json
"""
import os, sys, json
from pathlib import Path
import pandas as pd, numpy as np

# ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# env/config (can override with env vars)
PARQUET = os.getenv("PARQUET_PATH", "data/parquet_cotas/COTAHIST_A2025.parquet")
MIN_FIRST_PRICE = float(os.getenv("MIN_FIRST_PRICE", "1.0"))   # default 1.0
MIN_DAYS = int(os.getenv("MIN_DAYS", "10"))
MIN_VOLUME = float(os.getenv("MIN_VOLUME", "100"))
TOPN = int(os.getenv("TOPN", "10"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "200"))

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)

def sl(raw, start, end): return raw.str.slice(start-1, end).str.strip()

p = Path(PARQUET)
if not p.exists():
    print("Parquet não encontrado:", PARQUET); sys.exit(1)

# read parquet and get raw column
dfp = pd.read_parquet(p)
col = dfp.columns[0]
s = dfp[col].astype(str).fillna("")

# parse fixed-width into useful fields (COTAHIST layout)
df = pd.DataFrame({
 "TIPREG": sl(s,1,2), "DATA": sl(s,3,10), "CODNEG": sl(s,13,24),
 "PREULT": sl(s,109,121), "TOTNEG": sl(s,148,152), "QUATOT": sl(s,153,170)
})
df = df[df["TIPREG"]=="01"].copy()
df["DATA"] = pd.to_datetime(df["DATA"], format="%Y%m%d", errors="coerce")
# numeric conversions
df["PREULT"] = pd.to_numeric(df["PREULT"].str.replace(r'\D','',regex=True), errors="coerce")/100.0
df["QUATOT"] = pd.to_numeric(df["QUATOT"].str.replace(r'\D','',regex=True), errors="coerce")
df["TOTNEG"] = pd.to_numeric(df["TOTNEG"].str.replace(r'\D','',regex=True), errors="coerce")

# compute stats by ticker
tmp = df.dropna(subset=["CODNEG","DATA","PREULT"]).sort_values(["CODNEG","DATA"])
grouped = tmp.groupby("CODNEG").agg(
  first_price=("PREULT","first"),
  last_price=("PREULT","last"),
  days_count=("DATA","nunique"),
  total_trades=("TOTNEG","sum"),
  total_volume=("QUATOT","sum")
).reset_index()
grouped["rendimento_percent"] = (grouped["last_price"] - grouped["first_price"]) / grouped["first_price"] * 100.0

# apply filters
filt = (grouped["first_price"] >= MIN_FIRST_PRICE) & (grouped["last_price"] > 0) & \
       (grouped["days_count"] >= MIN_DAYS) & (grouped["total_volume"] >= MIN_VOLUME)
cleaned = grouped[filt].replace([np.inf, -np.inf], np.nan).dropna(subset=["rendimento_percent"]).sort_values("rendimento_percent", ascending=False).head(TOPN)

# save cleaned CSV
cleaned_csv = out_dir / "cleaned_top.csv"
cleaned.to_csv(cleaned_csv, index=False)
print("Saved:", cleaned_csv)

# build improved investigative prompt
lines = [f"{row['CODNEG']}: {row['rendimento_percent']:.2f}%" for _, row in cleaned.iterrows()]
prompt_header = (
    "Abaixo está uma lista de ativos (codigo B3) com seus rendimentos percentuais no período.\n"
    "Para cada um dos TOP 5, responda em Português com 3 itens:\n"
    "  1) possíveis causas para a valorização (ex.: corporate action, split, agrupamento, baixa liquidez, unidade, ETF, erro de dados);\n"
    "  2) quais métricas checar imediatamente (dias negociados, volume total, TOTNEG, notícias, comunicados);\n"
    "  3) um passo acionável para validar (ex.: excluir, investigar, incluir em watchlist, verificar comunicados).\n\n"
)
prompt_body = "\n".join(lines)
prompt = prompt_header + prompt_body

# save prompt
prompt_file = out_dir / "ai_prompt.txt"
prompt_file.write_text(prompt, encoding="utf-8")
print("Prompt salvo em:", prompt_file)

# attempt to call OpenAI if helper available
analyze = None
try:
    from openai_integration import analyze_text_with_openai
    analyze = analyze_text_with_openai
except Exception as e:
    try:
        # try importing as package src.openai_integration
        from src.openai_integration import analyze_text_with_openai
        analyze = analyze_text_with_openai
    except Exception as e2:
        analyze = None

if analyze is None:
    print("OpenAI helper não disponível. Se quiser o resumo real, configure OPENAI_API_KEY e crie src/openai_integration.py")
    print("Prompt (copy & paste to ChatGPT or save):\n", prompt[:1000], "...\n(arquivo completo em outputs/ai_prompt.txt)")
    sys.exit(0)

# call API and save outputs
print("Chamando OpenAI (modelo:", OPENAI_MODEL, ")...")
try:
    # allow override of model and max_tokens from env
    model = os.getenv("OPENAI_MODEL", OPENAI_MODEL)
    max_tokens = int(os.getenv("MAX_TOKENS", MAX_TOKENS))
    ai_text = analyze(prompt, model=model, max_tokens=max_tokens, save_raw_to=str(out_dir / "ai_raw.json"))
    # save ai results
    (out_dir / "ai_summary.txt").write_text(ai_text, encoding="utf-8")
    print("--- AI Summary ---\n", ai_text)
    print("AI summary salvo em:", out_dir / "ai_summary.txt")
except Exception as e:
    print("Erro ao chamar OpenAI:", e)
    # ensure prompt is available for manual copy
    print("Prompt salvo em outputs/ai_prompt.txt. Você pode colar no ChatGPT manualmente.")

#!/usr/bin/env python3
# src/local_parquet_ai.py
"""
Le um parquet (gerado a partir do COTAHIST), calcula retornos por ticker
(e.g. last vs first price), mostra os top N e pede um resumo à OpenAI.

Uso:
  python src/local_parquet_ai.py path/para/parquet [n]

Exemplo:
  python src/local_parquet_ai.py b3pulse-etl/data/parquet_cotas/COTAHIST_A2025.parquet 10
"""

import sys
from pathlib import Path
import pandas as pd

# garantir que openai_integration (helper) está importável a partir de src/
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from openai_integration import analyze_text_with_openai
except Exception as e:
    analyze_text_with_openai = None
    print("Aviso: não foi possível importar openai_integration (sem OpenAI). Erro:", e)
    print("O script ainda calculará os top N e imprimirá o prompt.")

def find_col(df, candidates):
    """
    Retorna o nome real da coluna presente no dataframe, tentando várias variações.
    """
    cols = [c for c in df.columns]
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    # tentativa por substring
    for c in cols:
        for cand in candidates:
            if cand.lower() in c.lower():
                return c
    return None

def compute_returns(df, ticker_col, close_col):
    """
    Calcula rendimento percentual (last - first) / first * 100 por ticker.
    """
    # tenta ordenar por data se existir
    date_col = find_col(df, ["date", "data", "dt", "timestamp"])
    if date_col:
        df_sorted = df.sort_values([ticker_col, date_col])
    else:
        # só ordena por ticker para garantir deterministicidade
        df_sorted = df.sort_values([ticker_col])
    first = df_sorted.groupby(ticker_col)[close_col].first()
    last = df_sorted.groupby(ticker_col)[close_col].last()
    rets = ((last - first) / first) * 100.0
    out = rets.reset_index().rename(columns={0: "rendimento_percent"})
    # ajustar nomes: garantir que o ticker fique como 'ticker'
    out.columns = [ticker_col, "rendimento_percent"]
    return out.sort_values("rendimento_percent", ascending=False)

def make_prompt(top_df, ticker_col):
    lines = [f"{row[ticker_col]}: {row['rendimento_percent']:.2f}%" for _, row in top_df.iterrows()]
    text_prompt = "Top {} ativos por rendimento:\n\n".format(len(top_df)) + "\n".join(lines)
    return text_prompt

def main(parquet_path, n=10):
    p = Path(parquet_path)
    if not p.exists():
        print("Parquet não encontrado:", parquet_path)
        sys.exit(1)

    print("Carregando parquet:", parquet_path)
    # pandas usa pyarrow por baixo para parquet; assegure pyarrow instalado
    try:
        df = pd.read_parquet(p)
    except Exception as e:
        print("Falha ao ler parquet com pandas:", e)
        sys.exit(1)

    print("Colunas detectadas (exemplo):", df.columns.tolist()[:40])

    # detectar colunas de ticker e preco/fechamento
    ticker_col = find_col(df, ["ticker", "codneg", "papel", "codigo", "ativo"])
    close_col = find_col(df, ["close", "fechamento", "preco_fechamento", "preco", "last", "vl_fechamento", "fech"])

    if ticker_col is None or close_col is None:
        print("Não foi possível identificar automaticamente as colunas ticker/close.")
        print("Colunas encontradas:", df.columns.tolist())
        print("Edite o script para especificar quais colunas usar.")
        sys.exit(1)

    print("Usando coluna ticker:", ticker_col, " e coluna de preço:", close_col)

    # converter colunas numéricas se necessário
    try:
        df[close_col] = pd.to_numeric(df[close_col], errors="coerce")
    except Exception:
        pass

    rets = compute_returns(df, ticker_col, close_col)
    topn = rets.head(int(n))
    prompt = make_prompt(topn, ticker_col)

    print("\n=== Preview Top {} ===".format(n))
    print(prompt)

    if analyze_text_with_openai is None:
        print("\nOpenAI helper não disponível. Instale openai_integration.py corretamente para gerar o resumo via API.")
        return

    try:
        print("\nSolicitando resumo à OpenAI...")
        summary = analyze_text_with_openai(
            "Em Português, gere 3 insights curtos e acionáveis sobre a lista abaixo:\n\n" + prompt,
            max_tokens=200
        )
        print("\n--- AI Summary ---")
        print(summary)
    except Exception as e:
        print("Erro ao chamar OpenAI:", e)
        print("Se quiser testar sem OpenAI, defina OPENAI_MOCK=1 (ou remova a dependência no script).")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/local_parquet_ai.py path/to/parquet [n]")
        sys.exit(1)
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10)

# from etl.extract import extract_zip
# from etl.transform import parse_b3_file
# from etl.load import save_to_parquet
from analysis.analytics import (
    show_top_returns_spark,
    show_bottom_returns_spark,
    join_and_analyze_spark
)
# from portfolio.manager import add_stock, remove_stock, show_portfolio

# --- OPENAI / capture helper (ADICIONE AQUI logo após os imports existentes) ---
import io
import sys
from openai_integration import analyze_text_with_openai

def capture_output(func, *args, **kwargs):
    """
    Executa func(*args, **kwargs) e captura tudo que for impresso (stdout).
    Retorna a string capturada.
    Útil para capturar o resultado de funções que usam DataFrame.show() ou prints.
    """
    old_stdout = sys.stdout
    try:
        buf = io.StringIO()
        sys.stdout = buf
        func(*args, **kwargs)
        return buf.getvalue()
    finally:
        sys.stdout = old_stdout

def ai_summary_menu():
    """
    Menu simples para gerar resumo via OpenAI dos relatórios de análise.
    Usa as funções existentes de analytics (que imprimem resultados) e captura a saída.
    """
    print("\n--- AI SUMMARY MENU ---")
    print("Escolha o relatório para resumir:")
    print("1 - Maiores rendimentos (show_top_returns_spark)")
    print("2 - Menores rendimentos (show_bottom_returns_spark)")
    print("3 - Comparar bases (join_and_analyze_spark)")
    opt = input("Opção (1/2/3): ").strip()

    if opt == "1":
        parquet_path = input("Caminho para parquet (ex: ./data/b3.parquet): ").strip()
        n = int(input("Quantos resultados (n): ").strip() or "10")
        # Captura a saída do show_top_returns_spark
        printed = capture_output(show_top_returns_spark, parquet_path, n)
        prompt = f"Below is the textual output of a table of top returns (ticker and rendimento). Give me 3 short insights (Portuguese):\n\n{printed}"
    elif opt == "2":
        parquet_path = input("Caminho para parquet (ex: ./data/b3.parquet): ").strip()
        n = int(input("Quantos resultados (n): ").strip() or "10")
        printed = capture_output(show_bottom_returns_spark, parquet_path, n)
        prompt = f"Below is the textual output of a table of bottom returns (ticker and rendimento). Give me 3 short insights (Portuguese):\n\n{printed}"
    elif opt == "3":
        p1 = input("Caminho parquet base1 (ex: ./data/2024.parquet): ").strip()
        p2 = input("Caminho parquet base2 (ex: ./data/2025.parquet): ").strip()
        n = int(input("Quantos resultados (n): ").strip() or "10")
        ordem = input("ordem (desc/asc) [desc]: ").strip() or "desc"
        printed = capture_output(join_and_analyze_spark, p1, p2, n, ordem)
        prompt = f"Below is the textual output of a comparison between two bases. Provide 3 short insights (Portuguese):\n\n{printed}"
    else:
        print("Opção inválida.")
        return

    # Chama OpenAI
    try:
        print("\nSolicitando resumo à OpenAI...")
        summary = analyze_text_with_openai(prompt, model="gpt-4o-mini", max_tokens=250, temperature=0.2)
        print("\n--- AI Summary ---")
        print(summary)
    except Exception as e:
        print("Falha ao chamar OpenAI:", e)


# def analysis_menu():
#     print("\n--- ANÁLISE ---")
#     print("1. Maiores rendimentos")
#     print("2. Menores rendimentos")
#     print("3. Ver rendimentos de uma base")
#     print("4. Comparar bases (join)")
#     print("0. Voltar")

# if __name__ == "__main__":
#     while True:
#         print("\n--- MENU ---")
#         print("1. Extrair arquivos B3")
#         print("2. Transformar dados de 2024 e 2025 com Pandas")
#         print("3. Análise de dados com SparkSQL")
#         print("4. Adicionar ação à favoritos")
#         print("5. Remover ação da favoritos")
#         print("6. Ver favoritos")
#         print("0. Sair")
#         op = input("Escolha: ")

#         if op == "1":
#             extract_zip("data/raw/COTAHIST_A2024.ZIP", "data/extracted/")
#             extract_zip("data/raw/COTAHIST_A2025.ZIP", "data/extracted/")
#             print("Arquivos extraídos.")

#         elif op == "2":
#             df_2024 = parse_b3_file("data/extracted/COTAHIST_A2024.TXT")
#             save_to_parquet(df_2024, "data/processed/b3_2024.parquet")
#             df_2025 = parse_b3_file("data/extracted/COTAHIST_A2025.TXT")
#             save_to_parquet(df_2025, "data/processed/b3_2025.parquet")
#             print("Dados de 2024 e 2025 transformados e salvos.")

#         elif op == "3":
#             while True:
#                 analysis_menu()
#                 a_op = input("Escolha análise: ")
#                 if a_op == "1":
#                     base = input("Qual base? (2024 ou 2025): ")
#                     n = int(input("Quantos resultados? "))
#                     parquet = f"data/processed/b3_{base}.parquet"
#                     show_top_returns_spark(parquet, n)
#                 elif a_op == "2":
#                     base = input("Qual base? (2024 ou 2025): ")
#                     n = int(input("Quantos resultados? "))
#                     parquet = f"data/processed/b3_{base}.parquet"
#                     show_bottom_returns_spark(parquet, n)
#                 elif a_op == "3":
#                     base = input("Qual base? (2024 ou 2025): ")
#                     n = int(input("Quantos resultados? "))
#                     parquet = f"data/processed/b3_{base}.parquet"
#                     show_top_returns_spark(parquet, n)
#                 elif a_op == "4":
#                     tipo = input("Comparar melhores ou piores? (m/p): ").lower()
#                     n = int(input("Quantos resultados comparar? "))
#                     ordem = "desc" if tipo == "m" else "asc"
#                     join_and_analyze_spark("data/processed/b3_2024.parquet", "data/processed/b3_2025.parquet", n, ordem)

#                 elif a_op == "0":
#                     break

#         elif op == "4":
#             ticker = input("Ticker: ").upper()
#             add_stock(ticker)

#         elif op == "5":
#             ticker = input("Ticker: ").upper()
#             remove_stock(ticker)

#         elif op == "6":
#             print("Sua carteira:", show_portfolio())

#         elif op == "0":
#             break
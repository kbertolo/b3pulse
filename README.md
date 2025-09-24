# B3-Pulse-ETL

Projeto para processar COTAHIST, calcular retornos e gerar resumos via OpenAI.

**Files of interest**
- `src/` - código
- `scripts/run_and_summarize.py` - pipeline local + chamada OpenAI
- `tools/parse_cotahist_from_parquet.py` - parser do COTAHIST
- `outputs/` - resultados (não comitados)

Antes de rodar, crie um virtualenv e instale dependências:

*pip install -r requirements.txt*
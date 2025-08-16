# conversational-trade-agent

This repo packages the notebook agent into modules. Use `app.ipynb` as your entrypoint.

To generate the synthetic data use `python gen_synthetic_data.py`

## Structure
- `trader_agent/` — Python package
  - `config.py` — toggles (OpenAI/HF), paths, dates
  - `data_store.py` — load CSV, build docs, FAISS vector index
  - `persona.py` — persona extraction from CSV
  - `date_utils.py` — human date range parsers
  - `scoring.py` — setup score heuristic
  - `intent.py` — LLM intent classifier (few-shot)
  - `lessons.py` — compute stats + LLM summarizer
  - `retriever.py` — filtered retrieval + scoring
  - `composer.py` — LLM response composer
  - `pipeline.py` — LangGraph orchestration
  - `bootstrap.py` — end-to-end wiring; exposes `boot_agent()`
- `app.ipynb` — demo notebook
- `synthetic_data.py` — Synthetic data generation
- `requirements.txt`

## Quick start
1. Install (if needed): `pip install -r requirements.txt`
2. Run (if required): `python gen_synthetic_data.py` to generate a synthetic data.
3. Put your CSV at `trader_past_trades.csv` or set `CSV_PATH` env var.
4. Set API keys if using OpenAI/HF.
5. Open `app.ipynb` and run all cells.

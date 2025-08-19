# Ask My Shipment

- Ask My Shipment is a Streamlit app for exploring UN Comtrade trade data using natural language queries. I only used a sample data for this as part of my informal learning of LLM applications
- It uses DuckDB for fast local analytics and OpenAI for code generation and insights.

## Features

- Query trade data with plain English questions
- See results as tables and interactive charts
- Download results (CSV, Vega-Lite JSON)
- On-demand LLM-generated insights
- Data dictionary and sample data preview

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/ask_my_shipment.git
cd ask_my_shipment
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Get the data

- Download UN Comtrade data (CSV or Parquet) for your countries/years of interest.
- Place files in `data/raw/` or `data/warehouse/`.
- Run the notebook or ingestion script to create `data/warehouse/ask_my_shipment.duckdb`.

### 4. Set your OpenAI API key

```bash
export OPENAI_API_KEY="sk-...your-key-here..."
```
Or add to `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-...your-key-here..."
```

### 5. Run the app

```bash
streamlit run streamlit_app/app.py
```

```
ask_my_shipment/
├── data/                # Data files (raw, processed, warehouse)
├── notebooks/           # Jupyter notebooks for EDA and ingestion and core logic
├── streamlit_app/       # Streamlit UI
│   └── app.py
├── utils/               # Core logic and helpers
│   ├── core.py
│   ├── helpers.py
│   ├── insights.py
│   └── db.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How to ingest/setup the database

- Use the provided notebook (`notebooks/02 - EDA.ipynb`) to clean and process raw data.
- The ingestion script/notebook will create the DuckDB file at `data/warehouse/ask_my_shipment.duckdb`.
- The main table is named `trade`.

## Data Source

- [UN Comtrade](https://comtradeplus.un.org/)
- Data dictionary: see `data_dictionary.csv` or preview in the app.


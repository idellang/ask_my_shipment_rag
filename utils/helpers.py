from pathlib import Path
import duckdb
import pandas as pd

from utils.db import get_ro_conn

# Constants (moved here)
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "warehouse" / "ask_my_shipment.duckdb"
DICT_PATH = ROOT / "data_dictionary.csv"


def get_sample_trade(limit: int = 10) -> pd.DataFrame:
    con = get_ro_conn()
    return con.execute(f"SELECT * FROM trade LIMIT {int(limit)}").fetchdf()


def _schema_text() -> str:
    con = get_ro_conn()
    return con.execute("DESCRIBE trade").fetchdf().to_string(index=False)


def get_schema_text() -> str:
    return _schema_text()


def _dict_text() -> str:
    if DICT_PATH.exists():
        df = pd.read_csv(DICT_PATH)
        return df.head(40).to_string(index=False)
    return "No data dictionary available."


def get_dict_text() -> str:
    return _dict_text()


def get_duckdb_conn():
    return get_ro_conn()

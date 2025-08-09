from pathlib import Path
import glob
import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "warehouse" / "ask_my_shipment.duckdb"

FINAL_COLS = [
    "year",
    "reporter",
    "import_export",
    "flowdesc",
    "partner",
    "isgrosswgtestimated",
    "fobvalue",
]

PARQUET_GLOBS = [
    (ROOT / "data" / "processed" / "*.parquet").as_posix(),
    (ROOT / "data" / "raw" / "*.parquet").as_posix(),
]


def _first_match(globs):
    for pat in globs:
        files = sorted(glob.glob(pat))
        if files:
            return files[0]
    return None


def _create_empty_table(con):
    con.execute(
        """
        CREATE TABLE trade (
            year INTEGER,
            reporter VARCHAR,
            import_export VARCHAR,
            flowdesc VARCHAR,
            partner VARCHAR,
            isgrosswgtestimated DOUBLE,  -- stays DOUBLE if it's 0/1 or numeric
            fobvalue DOUBLE
        )
        """
    )


def _ingest_files_any(p: str):
    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
    else:
        df = pd.read_csv(p)

    keep = [c for c in FINAL_COLS if c in df.columns]
    return df[keep] if keep else pd.DataFrame(columns=FINAL_COLS)


def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DB_PATH.exists():
        return

    con = duckdb.connect(str(DB_PATH))

    try:
        path = _first_match(PARQUET_GLOBS)
        if path:
            df = _ingest_files_any(path)
            if df.empty:
                _create_empty_table(con)
            else:
                con.register("df", df)
                con.execute("CREATE TABLE trade AS SELECT * FROM df")
        else:
            _create_empty_table(con)
        con.execute("CHECKPOINT")
    finally:
        con.close()


def get_ro_conn():
    ensure_db()
    return duckdb.connect(str(DB_PATH), read_only=True)

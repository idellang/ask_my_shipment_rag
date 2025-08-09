import utils.bootstrap_env  # isort: skip

# prioritize above for streamlit to load env vars first
import streamlit as st
import pandas as pd
import altair as alt
import json

from utils.db import get_ro_conn
from utils.core import answer
from utils.insights import generate_llm_insights
from utils.helpers import (
    get_sample_trade, get_schema_text, get_dict_text,
    ROOT, DB_PATH, DICT_PATH, _schema_text, _dict_text
)
from utils.keys import get_openai_key

# top page
st.set_page_config(
    page_title="Ask My Shipment",
    page_icon=":shipit:",
    layout="wide",
)
st.caption("Explore trade data powered by DuckDB + LLM code generation.")


# initialize session state
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

# read local if present
OPENAI_API_KEY = get_openai_key()
if not OPENAI_API_KEY:
    st.warning("OpenAI API key not set. Add it to .streamlit/secrets.toml or as an environment variable `OPENAI_API_KEY` to enable LLM features.")


@st.cache_resource
def cached_conn():
    return get_ro_conn()


@st.cache_data(ttl=600)
def schema_text():
    con = cached_conn()
    return con.execute("DESCRIBE trade").fetchdf().to_string(index=False)


@st.cache_data(ttl=600)
def sample_rows(n=50):
    con = cached_conn()
    return con.execute(f"SELECT * FROM trade LIMIT {n}").fetchdf()


if st.button("Refresh schema & data cache"):
    st.cache_data.clear()


# about
with st.expander("About this app", expanded=True):
    st.markdown(f"""
    - This app lets you ask questions about a local duckDB called 'trade'
    - Data Source: UN Comtrade (processed locally in this repo)
    - Duck DB Path: `{DB_PATH}`
    """)
    st.markdown(
        f"""
        [UN Comtrade](https://comtrade.un.org/) is a global trade database maintained by the United Nations.
        It provides detailed trade statistics and data on international trade flows.
"""
    )

st.subheader("Sample data and data dictionary")

cols = st.columns(2)
with cols[0]:
    st.markdown("### Sample Trade Data")
    try:
        st.dataframe(get_sample_trade(
            10), use_container_width=True, height=300)
    except Exception as e:
        st.error(f"Error loading sample trade data: {e}")

with cols[1]:
    st.markdown("### Data Dictionary")
    try:
        dict_text = get_dict_text()
        if dict_text:
            st.code(dict_text, language="markdown")
        else:
            st.warning("No data dictionary available.")
    except Exception as e:
        st.error(f"Error loading data dictionary: {e}")

# query input
st.subheader("Ask a question about the trade data")
q = st.text_area("Your question about the trade dataset",
                 height=80,
                 placeholder="e.g., Top 5 partner countries for China by FOB value",
                 key="query_input",
                 )

if st.button("Run Query", key="run_btn"):
    if not q.strip():
        st.warning("Please enter a question to run the query.")
    else:
        with st.spinner("Generating code and running query..."):
            try:
                result = answer(q)
            except Exception as e:
                st.error(f"Error processing your question: {e}")
                result = {"df": pd.DataFrame(), "chart": None, "chart_spec": None,
                          "code": "", "explanation": f"Error: {e}"}

        # persist last result
        st.session_state.last_result = result
        st.session_state.last_query = q

# render results
result = st.session_state.last_result
if result:
    st.subheader("Query Result")

    df = result.get("df")
    chart = result.get("chart", None)
    chart_spec = result.get("chart_spec", None)

    st.caption("First 10 rows of the result:")
    st.dataframe(df.head(10), use_container_width=True, height=300)

    with st.expander("Full DataFrame", expanded=False):
        st.dataframe(df, use_container_width=True, height=500)

    if chart is not None:
        st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("No chart generated for this query.")

    st.subheader("Downloads: Note this might not work in Streamlit Cloud")
    dl_cols = st.columns(3)

    with dl_cols[0]:
        st.download_button(
            "Download DataFrame as CSV",
            df.to_csv(index=False),
            file_name="trade_query_result.csv",
            mime="text/csv"
        )
    with dl_cols[1]:
        chart_json = json.dumps(chart_spec) if chart_spec else "{}"
        st.download_button(
            "Download Chart Spec as JSON",
            chart_json,
            file_name="chart_spec.json",
            mime="application/json"
        )

    with st.expander("See generated code and explanation"):
        st.code(result.get("code", ""), language="python")
        st.markdown(result.get("explanation", ""))

    # llm insights
    if st.button("Generate Insights", key="insights_btn"):
        with st.spinner("Generating insights..."):
            bullets = generate_llm_insights(
                result, st.session_state.last_query, max_rows=20, max_cols=12, model="gpt-4o")
        st.markdown(
            "\n".join(f"- {b}" for b in bullets), unsafe_allow_html=True)

else:
    st.caption("Run a query to see results here.")

st.caption("Tip: Set your OpenAI API key in the environment variable `OPENAI_API_KEY` to enable LLM features.")

import os
from dotenv import load_dotenv
load_dotenv(override=False)

try:
    import streamlit as st
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        os.environ.setdefault("OPENAI_API_KEY", st.secrets["OPENAI_API_KEY"])
except Exception:
    pass

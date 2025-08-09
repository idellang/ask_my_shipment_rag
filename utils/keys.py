import os
import streamlit as st


def get_openai_key():
    # Prefer Streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    # Fallback to environment variable
    return os.getenv("OPENAI_API_KEY")

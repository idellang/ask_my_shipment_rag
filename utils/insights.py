import json
import pandas as pd
from openai import OpenAI
from utils.keys import get_openai_key
from utils.core import LLM_MODEL


def _summarize_df_for_llm(df: pd.DataFrame, max_rows=20, max_cols=12):
    if df is None or df.empty:
        return {"schema": "empty", "sample_csv": "", "stats_csv": ""}
    cols = list(df.columns)[:max_cols]
    schema = "\n".join([f"- {c}: {str(df[c].dtype)}" for c in cols])
    sample_csv = df[cols].head(max_rows).to_csv(index=False)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    stats_csv = df[num_cols].describe().round(3).to_csv() if num_cols else ""
    return {"schema": schema, "sample_csv": sample_csv, "stats_csv": stats_csv}


def _to_chart_spec(chart_like):
    try:
        if chart_like is None:
            return None
        if isinstance(chart_like, dict):
            return chart_like
        if hasattr(chart_like, "to_dict"):
            return chart_like.to_dict()
    except Exception:
        pass
    return None


def _client():
    key = get_openai_key()
    if not key:
        raise ValueError(
            "OpenAI API key not set. Please configure it in .streamlit/secrets.toml or as an environment variable `OPENAI_API_KEY`.")
    return OpenAI(api_key=key)


def generate_llm_insights(results: dict, question: str, max_rows=20, max_cols=12, model=LLM_MODEL) -> list[str]:
    client = _client()
    df = results.get("df")
    snap = _summarize_df_for_llm(df, max_rows=max_rows, max_cols=max_cols)
    chart_spec = results.get(
        "chart_spec") or _to_chart_spec(results.get("chart"))
    chart_spec_str = json.dumps(chart_spec) if isinstance(
        chart_spec, dict) else "{}"

    system_msg = (
        "You are a concise data analyst. Given a small data snapshot and an optional chart spec, "
        "write 3-6 brief, business-friendly insights. Do not invent fields. "
        "Output plain text with each bullet starting with '- '."
    )
    user_msg = f"""
Question:
{question}

Data schema:
{snap['schema']}

Sample data (CSV):
{snap['sample_csv']}

Numeric summary:
{snap['stats_csv']}

Chart spec (Vega-Lite JSON):
{chart_spec_str}
"""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_msg},
                      {"role": "user", "content": user_msg}],
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        bullets = []
        for ln in lines:
            if ln.startswith("- "):
                bullets.append(ln)
            elif ln.startswith("* "):
                bullets.append("- " + ln[2:])
        if not bullets and text:
            bullets = ["- " + text]
        return bullets[:6] or ["- No insights available."]
    except OpenAI.RateLimitError:
        return ["- Insight generation failed: OpenAI API rate limit exceeded. Please try again later."]
    except OpenAI.AuthenticationError:
        return ["- Insight generation failed: Invalid OpenAI API key. Please check your secrets configuration."]
    except OpenAI.APIError as e:
        return [f"- Insight generation failed: OpenAI API returned an error: {e}"]
    except Exception as e:
        return [f"- Insight generation failed with an unexpected error: {e}"]

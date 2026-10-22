from pathlib import Path
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from data_pipeline import collect_laptops, load_data, DATA_PATH
from ml import train_models, predict_price, FEATURES

load_dotenv()
st.set_page_config(page_title="DataForge AI", page_icon="⚒️", layout="wide")

st.title("⚒️ DataForge AI")
st.caption("From the live web to a validated ML model — powered by TinyFish")
st.markdown(
    "Describe the market question, collect public listing evidence, validate a dataset, "
    "train measurable models, and predict with a traceable source trail."
)

with st.sidebar:
    st.header("TinyFish connection")
    if os.getenv("TINYFISH_API_KEY"):
        st.success("API key found in environment")
    else:
        st.warning("API key not found")
    st.caption("Your API key stays in your local .env file. Never paste it into the app or commit it.")

tab_collect, tab_data, tab_train, tab_predict = st.tabs(
    ["1 · Collect live data", "2 · Inspect dataset", "3 · Train & evaluate", "4 · Predict"]
)

with tab_collect:
    st.subheader("Collect used gaming-laptop listings")
    query = st.text_input(
        "Natural-language data request",
        value="used gaming laptop RTX 3050 RTX 4050 16GB RAM price India",
    )
    source_limit = st.slider("Search result pages to inspect", min_value=2, max_value=10, value=5)
    st.info(
        "Routing: TinyFish Search discovers URLs → Fetch reads accessible page content → "
        "Agent extracts structured listings when a page looks relevant."
    )
    if st.button("Search and collect listings", type="primary"):
        if not os.getenv("TINYFISH_API_KEY"):
            st.error("Missing TINYFISH_API_KEY. Follow README setup, then restart Streamlit.")
        else:
            status = st.empty()
            try:
                df, errors = collect_laptops(query, max_sources=source_limit, progress=status.info)
                status.empty()
                st.success(f"Collection finished. Dataset currently contains {len(df)} unique rows.")
                if errors:
                    with st.expander(f"Collection notes ({len(errors)})"):
                        for err in errors:
                            st.write(f"- {err}")
                st.dataframe(df.tail(30), use_container_width=True)
                st.download_button("Download CSV", df.to_csv(index=False), "laptop_listings.csv", "text/csv")
            except Exception as exc:
                status.empty()
                st.error(f"Collection failed: {exc}")
                st.exception(exc)

with tab_data:
    st.subheader("Dataset and quality checks")
    df = load_data()
    a, b, c, d = st.columns(4)
    a.metric("Rows", len(df))
    b.metric("Unique source URLs", int(df["source_url"].nunique()) if len(df) and "source_url" in df else 0)
    c.metric("Valid prices", int(pd.to_numeric(df.get("price_inr", pd.Series(dtype=float)), errors="coerce").notna().sum()))
    d.metric("Missing GPU", int(df["gpu"].isna().sum() + (df["gpu"] == "Unknown").sum()) if len(df) and "gpu" in df else 0)
    if df.empty:
        st.warning("No listings yet. Use the Collect tab to gather live data.")
    else:
        st.dataframe(df, use_container_width=True)
        st.subheader("Missing values by field")
        st.dataframe(df.isna().sum().rename("missing_rows").to_frame(), use_container_width=True)
        st.download_button("Download dataset CSV", df.to_csv(index=False), "laptop_listings.csv", "text/csv")

with tab_train:
    st.subheader("Train models and measure performance")
    df = load_data()
    st.write("Baseline: Ridge Regression. Stronger candidate: Extra Trees. The model with lower hold-out MAE is selected.")
    if st.button("Train and evaluate", type="primary"):
        try:
            with st.spinner("Training and evaluating models…"):
                result = train_models(df)
            st.success(f"Selected model: {result['best_model']}")
            st.metric("Training rows", result["rows"])
            st.dataframe(pd.DataFrame(result["reports"]).T, use_container_width=True)
            st.caption("Metrics use a held-out test split; CV MAE is added when enough rows exist. Small/noisy datasets can produce unstable scores.")
        except Exception as exc:
            st.error(str(exc))
    metrics_path = Path("models/metrics.json")
    if metrics_path.exists():
        import json
        metrics = json.loads(metrics_path.read_text())
        st.subheader("Latest saved evaluation")
        st.write(f"Selected model: **{metrics['best_model']}** · Rows: **{metrics['rows']}**")
        st.dataframe(pd.DataFrame(metrics["reports"]).T, use_container_width=True)

with tab_predict:
    st.subheader("Estimate a used laptop's asking-price range")
    st.caption("This is a model estimate from collected asking prices, not a verified sale price or professional appraisal.")
    col1, col2 = st.columns(2)
    with col1:
        brand = st.selectbox("Brand", ["HP", "Lenovo", "Acer", "ASUS", "Dell", "MSI", "Gigabyte", "Other"])
        cpu = st.text_input("CPU", "Intel Core i7 12th Gen")
        gpu = st.text_input("GPU", "NVIDIA RTX 4050")
        condition = st.selectbox("Condition", ["Excellent", "Good", "Fair", "Poor", "Unknown"])
    with col2:
        ram = st.selectbox("RAM (GB)", [8, 16, 24, 32, 64], index=1)
        storage = st.selectbox("Storage (GB)", [256, 512, 1024, 2048], index=1)
        age = st.number_input("Age in years", min_value=0.0, max_value=15.0, value=2.0, step=0.5)
    if st.button("Predict fair price", type="primary"):
        try:
            estimate = predict_price({
                "brand": brand, "cpu": cpu, "gpu": gpu, "ram_gb": ram,
                "storage_gb": storage, "age_years": age, "condition": condition,
            })
            st.metric("Estimated price", f"₹{estimate:,.0f}")
            st.caption("Use the live listings in the Dataset tab as supporting evidence. Prediction reliability depends on dataset size and similarity to this laptop.")
            df = load_data()
            if not df.empty and "price_inr" in df:
                st.subheader("Recent collected evidence")
                cols = [c for c in ["model", "cpu", "gpu", "ram_gb", "storage_gb", "price_inr", "source_url"] if c in df]
                st.dataframe(df[cols].tail(10), use_container_width=True)
        except Exception as exc:
            st.error(f"{exc} Collect enough listings and train a model first.")

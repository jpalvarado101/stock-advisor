import streamlit as st
import requests


st.set_page_config(page_title="Trillion‑GBM Dashboard", layout="wide")


st.title("Trillion‑GBM Forecaster")


symbol = st.text_input("Symbol", "AAPL")
lookback = st.slider("Lookback (days)", 10, 90, 30)


if st.button("Train model"):
    with st.spinner("Training..."):
        r = requests.post("http://localhost:8000/train", json={"symbol": symbol, "lookback": lookback})
    if r.ok:
        m = r.json()["metrics"]
        st.success("Done!")
        st.line_chart(m["val_nll"], height=180)
        st.line_chart(m["val_mae"], height=180)
        st.line_chart(m["val_diracc"], height=180)
    else:
        st.error(r.text)
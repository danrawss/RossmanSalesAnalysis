import streamlit as st
import pandas as pd

st.set_page_config(page_title="Rossmann Sales Analysis", layout="wide")
st.title("📊 Rossmann Store Sales Analysis")

st.markdown("""
Welcome to the Rossmann Store Sales analysis dashboard!

**Datasets:**
- `train.csv`: Daily sales data per store  
- `store.csv`: Store metadata/features  
- `test.csv`: Submission template  
""")

st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records (train)", "1,017,209")
col2.metric("Number of Stores", "1,115")
col3.metric("Date Range", "2013-01-01 → 2015-07-31")

if st.checkbox("Show raw data sample"):
    df = pd.read_csv("data/train.csv", nrows=100)
    st.dataframe(df)

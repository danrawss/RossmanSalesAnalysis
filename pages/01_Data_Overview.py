import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.header("🔍 Data Overview")

@st.cache_data
def load_data():
    df_train = pd.read_csv("data/train.csv", parse_dates=["Date"])
    df_store = pd.read_csv("data/store.csv")
    # merge train & store
    df = df_train.merge(df_store, on="Store", how="left")
    return df

df = load_data()
st.write(f"**Combined data shape:** {df.shape}")

# 1️⃣ Missing / extreme values
st.subheader("Missing & Extreme Values")
missing = df.isna().sum()
st.table(missing[missing > 0])

# 2️⃣ Aggregation example
st.subheader("Weekly Sales Trend")
weekly = df.resample("W", on="Date").Sales.sum()
fig, ax = plt.subplots()
weekly.plot(ax=ax)
ax.set_xlabel("Week")
ax.set_ylabel("Sales")
st.pyplot(fig)

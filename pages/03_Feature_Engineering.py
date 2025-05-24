import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.header("⚙️ Feature Engineering")

@st.cache_data
def load_merged():
    df_train = pd.read_csv("data/train.csv", parse_dates=["Date"])
    df_store = pd.read_csv("data/store.csv")
    df = df_train.merge(df_store, on="Store", how="left")
    return df

df = load_merged()
st.write("Raw sample:", df.head())

# 1️⃣ Handling missing
st.subheader("Missing Value Imputation")
df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(), inplace=True)
st.write("Nulls left:", df.isna().sum().sum())

# 2️⃣ Encoding categorical
st.subheader("Encoding")
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cats = encoder.fit_transform(df[["StoreType", "Assortment"]])
df_enc = pd.DataFrame(cats, columns=encoder.get_feature_names_out(), index=df.index)
st.write(df_enc.head())

# 3️⃣ Scaling numerical
st.subheader("Scaling")
scaler = StandardScaler()
scaled = scaler.fit_transform(df[["CompetitionDistance", "Promo2SinceWeek"]].fillna(0))
df_scaled = pd.DataFrame(scaled, columns=["CompDist_z", "Promo2Week_z"], index=df.index)
st.write(df_scaled.head())

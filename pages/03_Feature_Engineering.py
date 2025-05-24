import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.data_utils import load_data

st.title("⚙️ Feature Engineering")
st.markdown("""
**Purpose of this page:**  
Transform raw Rossmann data into modeling-ready features by:
- Extracting date-based fields  
- Handling missing & extreme values  
- Encoding categorical variables  
- Scaling numeric variables  
- Creating aggregation-based features  
""")

@st.cache_data
def get_data():
    df = load_data()
    return df.copy()

df = get_data()

# Date-based features
st.subheader("📅 Date-Based Features")
df["year"]         = df["Date"].dt.year
df["month"]        = df["Date"].dt.month
df["day_of_week"]  = df["Date"].dt.dayofweek
df["week_of_year"] = df["Date"].dt.isocalendar().week

st.write("Sample of new date features:", df[["Date","year","month","day_of_week","week_of_year"]].head())

# Map month numbers to names
df["month_name"] = df["Date"].dt.month_name().str.slice(stop=3)  # e.g. "Jan", "Feb", …

# Recompute average sales by month name
monthly_sales = (
    df.groupby("month_name")
      .Sales.mean()
      # re-order the categories Jan → Dec
      .reindex([
         "Jan","Feb","Mar","Apr","May","Jun",
         "Jul","Aug","Sep","Oct","Nov","Dec"
      ])
      .reset_index()
)

fig_month = px.bar(
    monthly_sales,
    x="month_name", y="Sales",
    title="Average Sales by Month",
    labels={"month_name":"Month","Sales":"Avg Sales"}
)
st.plotly_chart(fig_month, use_container_width=True)
st.markdown("""
> **Why?**  
Annual seasonality is key in retail—this chart shows which months are strongest for sales.
""")

# Handling missing & extreme values
st.subheader("⚠️ Missing & Extreme Values")
mv = df[["CompetitionDistance","Promo2SinceWeek","Promo2SinceYear"]].isnull().sum()
st.write("Missing before imputation:", mv)

# Impute missing
df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(), inplace=True)
df["Promo2SinceWeek"].fillna(0, inplace=True)
df["Promo2SinceYear"].fillna(df["Promo2SinceYear"].median(), inplace=True)

st.write("Missing after imputation:", df[["CompetitionDistance","Promo2SinceWeek","Promo2SinceYear"]].isnull().sum())

# Distribution of CompetitionDistance
fig_comp = px.histogram(
    df, x="CompetitionDistance", nbins=50,
    title="Competition Distance After Imputation"
)
st.plotly_chart(fig_comp, use_container_width=True)
st.markdown("""
> **Note:**  
We imputed distances with the median to preserve central tendency; zeroed out missing promo fields, as “no promo” makes sense as 0.
""")

# Encoding categorical variables
st.subheader("🔠 One-Hot Encoding of Categories")
categoricals = ["StoreType","Assortment","PromoInterval"]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = encoder.fit_transform(df[categoricals])
encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(categoricals),
    index=df.index
)
st.write("Encoded feature sample:", encoded_df.head())
st.markdown("""
> **Why?**  
Converts text categories into binary columns so most ML models can ingest them directly.
""")

# Scaling numeric variables
st.subheader("📏 Scaling Numeric Features")
to_scale = ["CompetitionDistance","Promo2SinceWeek"]
scaler = StandardScaler()
scaled = scaler.fit_transform(df[to_scale])
scaled_df = pd.DataFrame(
    scaled,
    columns=[f"{c}_scaled" for c in to_scale],
    index=df.index
)
st.write("Scaled feature sample:", scaled_df.head())

# Faceted histogram of scaled features
melted = scaled_df.melt(var_name="feature", value_name="value")
fig_scaled = px.histogram(
    melted, x="value", facet_col="feature",
    title="Scaled Feature Distributions",
    nbins=50
)
st.plotly_chart(fig_scaled, use_container_width=True)
st.markdown("""
> **Why?**  
Standardizing to zero mean/unit variance accelerates and stabilizes model training.
""")

# Aggregation-based feature: avg sales by StoreType
st.subheader("📊 Aggregation Feature: Avg Sales per StoreType")
agg = df.groupby("StoreType").Sales.mean().reset_index()
fig_agg = px.bar(
    agg, x="StoreType", y="Sales",
    title="Avg Sales by StoreType",
    labels={"Sales":"Avg Sales"}
)
st.plotly_chart(fig_agg, use_container_width=True)
st.markdown("""
> **Why?**  
Captures group-level effects: e.g. “Type A” stores may systematically sell more than “Type C.”
""")

# (Bonus) Feature: months since competition opened
st.subheader("⏳ Competition Open Duration")
# derive months open = (Year ×12 + Month) – (CompetitionOpenSinceYear×12 + CompetitionOpenSinceMonth)
df["CompOpenMonths"] = (
    (df.year * 12 + df.month)
  - (df.CompetitionOpenSinceYear.fillna(df.year).astype(int) * 12
     + df.CompetitionOpenSinceMonth.fillna(df.month).astype(int))
)
fig_co = px.histogram(
    df, x="CompOpenMonths", nbins=30,
    title="Distribution of Months Since Competition Opened"
)
st.plotly_chart(fig_co, use_container_width=True)
st.markdown("""
> **Why?**  
Stores with longer-standing nearby competition might see lower sales—this feature quantifies that effect.
""")

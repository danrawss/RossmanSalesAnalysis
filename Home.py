import streamlit as st
import pandas as pd

st.set_page_config(page_title="Rossmann Sales Analysis", layout="wide")

st.title("📊 Rossmann Store Sales Analysis Dashboard")
st.markdown("""
**🎯 Project Overview**  
Rossmann is one of Europe’s largest drugstore chains, operating over **4,000** stores across more than a dozen countries.  
Since Germany is Rossmann’s primary market, this app focuses on **German** store performance and sales trends.  

**🔗 Data Sources**  
- **Train (`train.csv`)**: Daily sales per store (Jan 2013 – Jul 2015)  
  - Key columns: `Sales`, `Customers`, `Open`, `Promo`, `SchoolHoliday`  
- **Store (`store.csv`)**: Store metadata  
  - Features include: `StoreType`, `Assortment`, `CompetitionDistance`, `Promo2` details  
- **Test (`test.csv`)**: Submission format for forecasting  

**📚 Project Goals**  
1. **Data Overview** – 📈 Explore raw data, missing values, and core sales trends.  
2. **Geospatial Analysis** – 🗺️ Map and cluster stores by performance across Germany.  
3. **Feature Engineering** – 🛠️ Create date, competition, and promo features; encode & scale variables.  
4. **Modeling** – 🤖 Use clustering & logistic regression to find patterns and predict high‐sales days.  
5. **Results** – 📊 Fit a multiple regression, interpret coefficients, and evaluate model fit.

**🔗 Kaggle Dataset**  
Download the original data here:  
https://www.kaggle.com/c/rossmann-store-sales/data  
""")

# Quick high-level KPIs
@st.cache_data
def load_sample():
    df = pd.read_csv("data/train.csv", parse_dates=["Date"])
    return df

df = load_sample()
# Quick high-level KPIs (wider first column)
col1, col2, col3 = st.columns([2, 1, 1])
col1.metric("🗓️ Date Range", f"{df.Date.min().date()} → {df.Date.max().date()}")
col2.metric("🏬 Number of Stores", f"{df.Store.nunique():,}")
col3.metric("📈 Total Records",      f"{len(df):,}")

st.markdown("---")
st.markdown("👉 Use the sidebar to navigate through each analysis step. Adjust parameters interactively to see real-time updates and insights! 🚀")

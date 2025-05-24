import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🔍 Data Overview")
st.markdown("""
**Purpose of this page:**  
Load and inspect the Rossmann datasets to understand their structure,  
highlight missing or extreme values, and interactively explore key sales trends.
""")

@st.cache_data
def load_data():
    df_train = pd.read_csv("data/train.csv", parse_dates=["Date"])
    df_store = pd.read_csv("data/store.csv")
    return df_train, df_store

df_train, df_store = load_data()

# Dataset shapes
st.subheader("📦 Dataset Shapes")
st.markdown(f"- **Train:** {df_train.shape[0]:,} rows × {df_train.shape[1]} cols - **Store:** {df_store.shape[0]:,} rows × {df_store.shape[1]} cols")

# Raw samples
if st.checkbox("Show raw data samples"):
    st.write("**Train sample:**")
    st.dataframe(df_train.head())
    st.write("**Store sample:**")
    st.dataframe(df_store.head())

# Missing values
st.subheader("❓ Missing Values")
col1, col2 = st.columns(2)
with col1:
    missing_train = df_train.isnull().sum().loc[lambda x: x>0]
    st.markdown("**Train:**")
    st.dataframe(missing_train)
with col2:
    missing_store = df_store.isnull().sum().loc[lambda x: x>0]
    st.markdown("**Store:**")
    st.dataframe(missing_store)
st.markdown("*We’ll need to impute or flag these missing values before modeling.*")

# Interactive Sales Distribution
st.subheader("📊 Sales Distribution")
bins = st.slider("Number of histogram bins", 10, 200, 50)
fig_hist = px.histogram(
    df_train, x="Sales", nbins=bins, 
    log_y=True, title="Daily Sales Distribution (log y)",
    labels={"Sales":"Daily Sales"},
    hover_data=["Customers"]
)
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown("""
- **Why log y-axis?** It lets us see both the very common low-sales days and the rare extremely high-sales outliers on the same chart.  
- **Hover data:** Shows customer counts for each bin to understand the sales-customers relationship.
""")

# Interactive Boxplot for Outliers
st.subheader("🗃️ Boxplot of Daily Sales")
fig_box = px.box(
    df_train, x="Sales", 
    title="Boxplot of Daily Sales",
    labels={"Sales":"Daily Sales"},
)
st.plotly_chart(fig_box, use_container_width=True)
st.markdown("""
- **Boxplot interpretation:**  
  - The box spans the interquartile range (Q1 to Q3).  
  - The line in the box is the median daily sales.  
  - “Whiskers” extend to 1.5×IQR, beyond which points are considered outliers.
""")

# Monthly Sales Trend with date-range filter
st.subheader("📈 Monthly Sales Trend")
min_date, max_date = st.date_input(
    "Select date range", 
    value=(df_train.Date.min(), df_train.Date.max()),
    key="date_range"
)
mask = (df_train.Date >= pd.to_datetime(min_date)) & (df_train.Date <= pd.to_datetime(max_date))
monthly = (
    df_train.loc[mask]
            .set_index("Date")
            .Sales
            .resample("M")
            .sum()
            .reset_index()
)
fig_line = px.line(
    monthly, x="Date", y="Sales",
    title="Total Sales per Month",
    labels={"Sales":"Total Sales", "Date":"Month"},
    markers=True
)
st.plotly_chart(fig_line, use_container_width=True)
st.markdown("""
- **Seasonality:** Look for regular peaks (e.g., around holidays) or troughs in the year.  
- **Trend:** Observe if overall sales are increasing, flat, or declining over time.  
- **Date filter:** Allows focusing on specific periods (e.g., pre-promo campaigns).
""")

# Correlation of numeric features (with CompetitionDistance merged)
st.subheader("🔗 Correlation Between Numeric Features")
df_merged = df_train.merge(
    df_store[["Store", "CompetitionDistance"]], on="Store", how="left"
).copy()
df_merged["CompetitionDistance"].fillna(df_merged.CompetitionDistance.median(), inplace=True)

numeric_cols = ["Sales", "Customers", "CompetitionDistance"]
corr = df_merged[numeric_cols].corr().round(2)
fig_corr = px.imshow(
    corr,
    text_auto=True,
    aspect="auto",
    title="Correlation Matrix",
    labels=dict(x="Feature", y="Feature", color="Correlation")
)
st.plotly_chart(fig_corr, use_container_width=True)
st.markdown("""
- **Sales vs. Customers:** Strong positive correlation indicates higher footfall drives sales.  
- **Sales vs. CompetitionDistance:** A mild negative correlation suggests stores closer to competitors may see slightly lower sales.  
- **Customers vs. CompetitionDistance:** Helps understand if competitor proximity affects store traffic.
""")

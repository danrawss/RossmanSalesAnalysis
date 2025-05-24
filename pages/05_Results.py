import streamlit as st
import pandas as pd
import statsmodels.formula.api as smf
import plotly.express as px

from scripts.data_utils import load_data

st.title("🏆 Results & Multiple Regression")
st.markdown("""
**Purpose of this page:**  
Use Statsmodels to perform a multiple regression of daily sales on Competition Distance,  
Promo flag and Customer count, then interpret coefficients and assess model fit.
""")

@st.cache_data
def fit_ols():
    # Load merged train+store data
    df = load_data()
    # Impute missing competition distances
    df["CompetitionDistance"].fillna(df["CompetitionDistance"].median(), inplace=True)
    # Fit OLS: Sales ~ CompetitionDistance + Promo + Customers
    model = smf.ols("Sales ~ CompetitionDistance + Promo + Customers", data=df).fit()
    return model, df

model, df_full = fit_ols()

# 1) Regression diagnostics (clearer table)
st.subheader("Regression Diagnostics")

# Manually gather the key metrics
metrics = {
    "No. Observations":    int(model.nobs),
    "R-squared":           model.rsquared,
    "Adj. R-squared":      model.rsquared_adj,
    "F-statistic":         model.fvalue,
    "Prob (F-statistic)":  model.f_pvalue,
    "Log-Likelihood":      model.llf,
    "AIC":                 model.aic,
    "BIC":                 model.bic,
}

# Build a tidy DataFrame
df_stats = (
    pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
      .assign(Value=lambda df: df.Value.round(3))
)

st.dataframe(df_stats, use_container_width=True)

st.markdown("""
- **No. Observations:** number of daily records used in the regression.  
- **R-squared** / **Adj. R-squared:** proportion of variance explained (adjusted for number of predictors).  
- **F-statistic** & **Prob (F-statistic):** test whether at least one predictor has nonzero coefficient.  
- **Log-Likelihood**, **AIC**, **BIC:** measures of model fit penalized for complexity.
""")

# 2) Coefficients table
st.subheader("Coefficient Estimates")
coefs = pd.DataFrame({
    "Coef.":   model.params,
    "Std Err": model.bse,
    "t-value": model.tvalues,
    "P>|t|":   model.pvalues
}).round(4)
st.dataframe(coefs, use_container_width=True)

st.markdown("""
- **Intercept:** expected sales when all predictors = 0.  
- **CompetitionDistance:** effect on sales per unit distance to nearest competitor.  
- **Promo:** average difference in sales on promotion vs. non-promotion days.  
- **Customers:** incremental sales per additional customer visit.  
""")

# 3) Actual vs. Predicted
st.subheader("Actual vs. Predicted Sales")
# Predict on the full dataset
df_full["Predicted"] = model.predict(df_full)
# Sample for speed
df_sample = df_full.sample(1000, random_state=42)

fig = px.scatter(
    df_sample, x="Sales", y="Predicted",
    trendline="ols",
    title="Actual vs. Predicted Sales (sample of 1,000)",
    labels={"Sales":"Actual Sales", "Predicted":"Predicted Sales"}
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("""
- The **scatter plot** shows how closely predicted values align with actual sales.  
- The **trend line** (in black) indicates the overall fit—points closer to it mean better predictions.  
- Deviations from the line highlight where the model under- or over-predicts.
""")

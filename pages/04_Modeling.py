import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import plotly.express as px
import plotly.figure_factory as ff

from scripts.data_utils import load_data, prepare_features

st.title("🧠 Modeling")
st.markdown("""
**Purpose of this page:**  
Build and evaluate simple ML models on Rossmann store data:  
1. **Clustering** stores by sales pattern  
2. **Logistic regression** to predict “high-sales” days  
3. Interactive metrics & visualizations for model interpretation  
""")

@st.cache_data
def get_features():
    df = load_data()
    X, y = prepare_features(df)
    return X, y

X, y = get_features()

# Clustering stores by sales
st.subheader("1️⃣ K-Means Clustering of Stores")
n_clusters = st.slider("Number of clusters", min_value=2, max_value=8, value=4)
km = KMeans(n_clusters=n_clusters, random_state=42)
clusters = km.fit_predict(X)
cluster_counts = pd.Series(clusters).value_counts().sort_index()

# Show cluster sizes bar chart
fig_clusters = px.bar(
    x=cluster_counts.index.astype(str),
    y=cluster_counts.values,
    labels={"x":"Cluster", "y":"Count"},
    title="Number of Stores per Cluster"
)
st.plotly_chart(fig_clusters, use_container_width=True)
st.markdown("""
- **What clustering does:** groups stores with similar feature profiles (competition distance & promo history).  
- **Interpretation:** cluster sizes show how many stores share each pattern.
""")

# Define “HighSales” and split data
median_sales = X["avg_sales"] if "avg_sales" in X.columns else None
st.subheader("2️⃣ Logistic Regression: Predict High-Sales Days")
# prepare binary target (already in prepare_features if you used that)
# but if not, we'll define:
y = y  # from prepare_features: 1 = high-sales day, 0 = otherwise

# train/test split slider
from sklearn.model_selection import train_test_split
test_size = st.slider("Test set proportion", 0.1, 0.5, 0.2, step=0.05)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# train logistic regression with regularization control
C = st.number_input("Inverse regularization (C)", min_value=0.01, max_value=10.0, value=1.0)
lr = LogisticRegression(C=C, max_iter=200)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]

# Metrics: classification report table
st.markdown("**Classification Report**")
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose().round(2)
st.dataframe(df_report)

st.markdown("""
- **Precision**: of days predicted high-sales, how many truly were.  
- **Recall**: of actual high-sales days, how many we caught.  
- **F1-score**: harmonic mean of precision & recall.
""")

# Confusion matrix heatmap
cm = confusion_matrix(y_test, y_pred)
fig_cm = ff.create_annotated_heatmap(
    z=cm,
    x=["Pred 0","Pred 1"],
    y=["True 0","True 1"],
    colorscale="Blues"
)
fig_cm.update_layout(title="Confusion Matrix", xaxis_title="", yaxis_title="")
st.plotly_chart(fig_cm, use_container_width=True)
st.markdown("Confusion matrix: rows = actual class, columns = predicted.")

# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
fig_roc = px.area(
    x=fpr, y=tpr,
    title=f"ROC Curve (AUC = {roc_auc:.2f})",
    labels=dict(x="False Positive Rate", y="True Positive Rate"),
    width=700, height=400
)
fig_roc.add_shape(
    type="line", line=dict(dash="dash"),
    x0=0, x1=1, y0=0, y1=1
)
st.plotly_chart(fig_roc, use_container_width=True)
st.markdown("""
- **ROC curve** shows trade-off between sensitivity and specificity across thresholds.  
- **AUC** (area under curve) quantifies overall separability (1.0 = perfect, 0.5 = random).
""")

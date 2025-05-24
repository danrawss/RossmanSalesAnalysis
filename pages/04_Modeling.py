import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scripts.data_utils import load_data, prepare_features

st.header("🧠 Modeling")

@st.cache_data
def get_data():
    df = load_data()
    return prepare_features(df)

X, y = get_data()

# 1️⃣ Clustering stores by competition distance
st.subheader("Clustering")
k = st.slider("Number of clusters", 2, 10, 4)
km = KMeans(n_clusters=k, random_state=42)
clusters = km.fit_predict(X)
st.write("Cluster counts:", pd.Series(clusters).value_counts())

# 2️⃣ Logistic regression
st.subheader("Logistic Regression")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
st.text(classification_report(y_test, y_pred))

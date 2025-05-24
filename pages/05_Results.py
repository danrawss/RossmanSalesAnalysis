import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from scripts.data_utils import load_data, prepare_features

st.header("🏆 Results & Evaluation")

@st.cache_data
def get_model_and_data():
    df = load_data()
    X, y = prepare_features(df)
    model = LogisticRegression(max_iter=200).fit(X, y)
    return model, X, y

model, X, y = get_model_and_data()

prob = model.predict_proba(X)[:, 1]
fpr, tpr, _ = roc_curve(y, prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
ax.plot([0,1], [0,1], "--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
st.pyplot(fig)

st.markdown(f"**AUC:** {roc_auc:.3f}")

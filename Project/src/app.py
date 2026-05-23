
import streamlit as st, pandas as pd, joblib, os, numpy as np
BASE = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE, 'models', 'iforest.joblib')


PROCESSED = os.path.join(BASE, 'data', 'processed', 'X_scaled.csv')

st.title('SIEM ML — Isolation Forest Demo')

if not os.path.exists(MODEL_PATH):
    st.error('Model not found. Train the model first by running src/train_iforest.py')
else:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED)
    scores = -model.decision_function(df.values)
    thr = float(np.quantile(scores, 1 - model.contamination))
    labels = (scores >= thr).astype(int)
    df_out = df.copy()
    df_out['anomaly_score'] = scores
    df_out['anomaly_label'] = labels
    st.subheader('Top anomalies')
    st.dataframe(df_out.sort_values('anomaly_score', ascending=False).head(10))
    st.subheader('Anomaly score distribution')
    st.bar_chart(pd.Series(scores).value_counts(bins=20).sort_index())

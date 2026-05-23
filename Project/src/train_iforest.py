
import os, joblib, pandas as pd
from sklearn.ensemble import IsolationForest

BASE = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE, 'data', 'processed')
MODEL_DIR = os.path.join(BASE, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'iforest.joblib')

def train_iforest(contamination=0.05, n_estimators=100, random_state=42):
    X_scaled = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_scaled.csv')).values
    model = IsolationForest(contamination=contamination, n_estimators=n_estimators, random_state=random_state)
    model.fit(X_scaled)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print('Model trained and saved to', MODEL_PATH)
    return model

if __name__ == '__main__':
    train_iforest()

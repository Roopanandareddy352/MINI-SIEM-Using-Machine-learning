
import os, joblib, pandas as pd, numpy as np
from sklearn.metrics import classification_report

BASE = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE, 'data', 'processed')
MODEL_PATH = os.path.join(BASE, 'models', 'iforest.joblib')

def evaluate():
    model = joblib.load(MODEL_PATH)
    X = pd.read_csv(os.path.join(PROCESSED_DIR, 'X_scaled.csv')).values
    y = pd.read_csv(os.path.join(PROCESSED_DIR, 'y.csv')).values.flatten()
    scores = -model.decision_function(X)
    thr = np.quantile(scores, 1 - model.contamination)
    preds = (scores >= thr).astype(int)
    y_bin = (y != 0).astype(int)
    print('Threshold:', thr)
    print(classification_report(y_bin, preds, digits=4))
    out = pd.DataFrame({'score': scores, 'pred': preds, 'label': y_bin})
    out.to_csv(os.path.join(PROCESSED_DIR, 'eval_results.csv'), index=False)
    print('Evaluation results saved to', os.path.join(PROCESSED_DIR, 'eval_results.csv'))

if __name__ == '__main__':
    evaluate()

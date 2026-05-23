
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import joblib

BASE = os.path.dirname(os.path.dirname(__file__))
DATA_IN = os.path.join(BASE, 'data', 'sample_dataset.csv')
PROCESSED_DIR = os.path.join(BASE, 'data', 'processed')
SCALER_PATH = os.path.join(BASE, 'models', 'scaler.joblib')

def load_and_clean(path=DATA_IN):
    df = pd.read_csv(path)
    if 'usb_mean_usb_dur' in df.columns:
        df['usb_mean_usb_dur'] = df['usb_mean_usb_dur'].replace(-1, 0.0)
    df = df.drop_duplicates().reset_index(drop=True)
    return df

def preprocess_and_save():
    df = load_and_clean()
    X = df.drop(columns=['insider'])
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    pd.DataFrame(Xs, columns=X.columns).to_csv(os.path.join(PROCESSED_DIR, 'X_scaled.csv'), index=False)
    df['insider'].to_csv(os.path.join(PROCESSED_DIR, 'y.csv'), index=False)
    print('Preprocessing done. Processed files saved to', PROCESSED_DIR)

if __name__ == '__main__':
    preprocess_and_save()

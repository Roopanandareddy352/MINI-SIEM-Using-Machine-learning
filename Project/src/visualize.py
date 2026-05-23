
import pandas as pd, os
BASE = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DIR = os.path.join(BASE, 'data', 'processed')

def show_top_anomalies(n=5):
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'eval_results.csv'))
    df_sorted = df.sort_values('score', ascending=False).head(n)
    print('Top anomalies (highest score first):')
    print(df_sorted)

if __name__ == '__main__':
    show_top_anomalies()

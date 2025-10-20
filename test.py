import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'dataset.csv'
MODEL_PATH = BASE_DIR / 'model.pkl'
COLS_PATH = BASE_DIR / 'model_columns.txt'
TEST_REPORT = BASE_DIR / 'test_report.json'


def main():
    if not DATA_PATH.exists() or not MODEL_PATH.exists() or not COLS_PATH.exists():
        raise SystemExit('dataset.csv, model.pkl, and model_columns.txt are required')

    df = pd.read_csv(DATA_PATH)
    with open(COLS_PATH, 'r', encoding='utf-8') as f:
        text_col = f.readline().strip()
    if text_col not in df.columns:
        raise ValueError(f'Text column {text_col} not found in dataset')

    model = joblib.load(MODEL_PATH)

    # ✅ Added 'fake' to label detection
    if 'label' in df.columns:
        y = df['label']
    elif 'target' in df.columns:
        y = df['target']
    elif 'fake' in df.columns:
        y = df['fake']
    else:
        raise ValueError('Label/target column not found to evaluate')

    y = y.map(lambda v: 1 if str(v).lower() in {'1', 'fake', 'fraud', 'scam', 'spam', 'true'} else 0)

    X = df[[text_col]].copy()
    y_pred = model.predict(X)

    acc = accuracy_score(y, y_pred)
    rep = classification_report(y, y_pred, output_dict=True)
    cm = confusion_matrix(y, y_pred).tolist()

    with open(TEST_REPORT, 'w', encoding='utf-8') as f:
        json.dump({'accuracy': acc, 'report': rep, 'confusion_matrix': cm}, f, indent=2)

    print(f'Wrote test report to {TEST_REPORT}. Accuracy: {acc:.4f}')


if __name__ == '__main__':
    main()


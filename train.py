import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'dataset.csv'
MODEL_PATH = BASE_DIR / 'model.pkl'
COLS_PATH = BASE_DIR / 'model_columns.txt'
REPORT_PATH = BASE_DIR / 'train_report.json'


def guess_columns(df: pd.DataFrame):
    text_col = None
    label_col = None
    lower_cols = {c.lower(): c for c in df.columns}
    
    # ✅ Added 'post_text' for your dataset
    for candidate in ['text', 'post', 'content', 'message', 'tweet', 'body', 'statement', 'description', 'post_text']:
        if candidate in lower_cols:
            text_col = lower_cols[candidate]
            break

    # ✅ Added 'fake' to label detection list (your dataset uses this)
    for candidate in ['label', 'target', 'is_fake', 'class', 'y', 'category', 'type', 'fake']:
        if candidate in lower_cols:
            label_col = lower_cols[candidate]
            break

    if text_col is None:
        raise ValueError('Could not find a text column (tried: text, post, content, message, tweet, body, statement, description, post_text)')
    if label_col is None:
        raise ValueError('Could not find a label column (tried: label, target, is_fake, class, y, category, type, fake)')

    print(f"✅ Detected text column: {text_col}")
    print(f"✅ Detected label column: {label_col}")
    return text_col, label_col


def normalize_labels(series: pd.Series) -> pd.Series:
    mapping = {
        'fake': 1, 'fraud': 1, 'scam': 1, 'spam': 1, '1': 1, 1: 1, True: 1,
        'real': 0, 'genuine': 0, 'legit': 0, '0': 0, 0: 0, False: 0
    }
    return series.map(lambda v: mapping.get(str(v).strip().lower(), v)).astype(int)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'dataset not found at {DATA_PATH}')
    
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print("✅ Dataset loaded successfully.")
    print("Columns in dataset:", df.columns.tolist())

    # Detect columns
    text_col, label_col = guess_columns(df)

    # Drop missing rows and normalize labels
    df = df.dropna(subset=[text_col, label_col]).copy()
    df[label_col] = normalize_labels(df[label_col])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df[[text_col]], df[label_col],
        test_size=0.2, random_state=42, stratify=df[label_col]
    )

    # TF-IDF text vectorization
    text_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1, 2)))
    ])

    # Preprocessing step
    preprocessor = ColumnTransformer([
        ('text', text_pipeline, text_col)
    ], remainder='drop')

    # Logistic Regression model pipeline
    model = Pipeline([
        ('cols', preprocessor),
        ('clf', LogisticRegression(max_iter=200))
    ])

    # Train model
    print("🚀 Training model...")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rep = classification_report(y_test, y_pred, output_dict=True)

    # Save model and metadata
    joblib.dump(model, MODEL_PATH)
    with open(COLS_PATH, 'w', encoding='utf-8') as f:
        f.write(text_col + '\n')

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump({'accuracy': acc, 'report': rep, 'text_col': text_col}, f, indent=2)

    print(f"✅ Trained model saved to {MODEL_PATH}")
    print(f"📊 Accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()


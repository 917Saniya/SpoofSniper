import os
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session

# ✅ Added for CORS
from flask_cors import CORS

import joblib
import pandas as pd
import numpy as np
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')

# ✅ Enable CORS for all routes (frontend JS can call Flask API)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'spoofsniper.db'
DATA_PATH = BASE_DIR / 'spoof_posts.csv'

# --------------------- Database ------------------------
engine = create_engine(f'sqlite:///{DB_PATH}', echo=False, future=True)
SessionLocal = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=False))
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(150), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(engine)

# --------------------- Model Section ------------------------
MODEL_PATH = BASE_DIR / 'model.pkl'
MODEL_COL_PATH = BASE_DIR / 'model_columns.txt'
_model = None
_vectorizer = None

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def train_model():
    if not DATA_PATH.exists():
        print("❌ spoof_posts.csv not found.")
        return None

    print("📘 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].apply(preprocess_text)

    X = df['text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Stronger text pipeline: TF-IDF (1,2) + balanced LogisticRegression
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=500, class_weight='balanced', n_jobs=None)
    model.fit(X_train_tfidf, y_train)

    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model trained successfully with accuracy: {acc:.3f}")

    joblib.dump((model, vectorizer), MODEL_PATH)
    with open(MODEL_COL_PATH, 'w', encoding='utf-8') as f:
        f.write('text\n')

    return model, vectorizer, acc

def load_model():
    global _model, _vectorizer
    if _model is None:
        if MODEL_PATH.exists():
            _model, _vectorizer = joblib.load(MODEL_PATH)
        else:
            print("⚙️ Training new model since none found...")
            _model, _vectorizer, _ = train_model()
    return _model, _vectorizer

# --------------------- Routes ------------------------
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/analysis')
def account_analysis():
    return render_template('analysis.html')

@app.route('/rules')
def detection_rules():
    return render_template('rules.html')

# Auth pages
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        email = request.form.get('email','').strip()
        if not username or not email:
            flash('Username and email required','danger')
            return render_template('login.html')
        # demo login (no password): create if not exists
        db = SessionLocal()
        user = db.query(User).filter((User.username==username)|(User.email==email)).first()
        if not user:
            user = User(username=username, email=email, password_hash='-')
            db.add(user)
            db.commit()
        session['user_id'] = user.id
        session['username'] = user.username
        flash('Logged in successfully','success')
        return redirect(url_for('landing'))
    return render_template('login.html')

@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username','').strip()
        email = request.form.get('email','').strip()
        if not username or not email:
            flash('Username and email required','danger')
            return render_template('signup.html')
        db = SessionLocal()
        if db.query(User).filter((User.username==username)|(User.email==email)).first():
            flash('User already exists','warning')
            return render_template('signup.html')
        user = User(username=username, email=email, password_hash='-')
        db.add(user)
        db.commit()
        flash('Account created. Please login.','success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out','info')
    return redirect(url_for('landing'))

# --------------------- API ------------------------

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        print("📩 Incoming /api/predict request")
        data = request.get_json(force=True)
        print("Received JSON:", data)

        # Validate input
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400

        text = data.get('text', '')
        username = data.get('username', '')
        followers = int(data.get('followers', 0))

        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400

        # Load model
        model, vectorizer = load_model()
        if model is None or vectorizer is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Preprocess and predict
        text_clean = preprocess_text(text)
        X_vec = vectorizer.transform([text_clean])
        pred = model.predict(X_vec)[0]
        proba = model.predict_proba(X_vec)[0]
        confidence = float(np.max(proba))

        # Explanation logic
        explanations = []
        text_lower = text.lower()
        urgency_words = ['urgent', 'emergency', 'act now', 'limited time', 'immediately', 'asap']
        money_words = ['money', 'cash', 'prize', 'winner', 'free', 'dollar', 'win', '$']
        action_words = ['click', 'verify', 'confirm', 'update', 'suspended', 'locked', 'hacked']

        for word in urgency_words + money_words + action_words:
            if word in text_lower:
                explanations.append(f"Keyword detected: '{word}'")

        # Risk scoring
        account_risk = 0
        if username and any(w in username.lower() for w in money_words):
            account_risk += 3
            explanations.append(f"Suspicious username: '{username}'")
        if followers < 50:
            account_risk += 1
            explanations.append(f"Low follower count: {followers}")

        text_risk = min(confidence * 100, 100) if pred == 'Fake' else 100 - (confidence * 100)
        overall_risk = min((text_risk + account_risk * 10) / 2, 100)

        # Final response
        return jsonify({
            'label': str(pred),
            'confidence': round(confidence, 3),
            'textRiskPct': round(text_risk, 1),
            'accountRiskPct': round(account_risk * 10, 1),
            'overallRiskPct': round(overall_risk, 1),
            'explanations': explanations[:5],
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'used_model': True,
            'metadata': {
                'username': username,
                'followers': followers,
                'text_length': len(text)
            }
        })

    except Exception as ex:
        print("❌ Prediction error:", str(ex))
        return jsonify({'error': str(ex)}), 500



# --------------------- Webhook Endpoint for n8n ------------------------
@app.route('/webhook/spoof-analysis', methods=['POST'])
def webhook_spoof_analysis():
    """Webhook endpoint for n8n integration"""
    try:
        model, vectorizer = load_model()
        data = request.get_json(force=True) or {}

        text = data.get('text', '')
        username = data.get('username', '')
        followers = data.get('followers', 0)

        if not text:
            return jsonify({'error': 'text is required'}), 400

        text_clean = preprocess_text(text)
        X_vec = vectorizer.transform([text_clean])
        pred = model.predict(X_vec)[0]
        proba = model.predict_proba(X_vec)[0]
        confidence = float(np.max(proba))

        explanations = []
        text_lower = text.lower()

        urgency_words = ['urgent', 'emergency', 'act now', 'limited time', 'immediately', 'asap']
        money_words = ['money', 'cash', 'prize', 'winner', 'free', 'dollar', 'win', '$']
        action_words = ['click', 'verify', 'confirm', 'update', 'suspended', 'locked', 'hacked']

        for word in urgency_words:
            if word in text_lower:
                explanations.append(f"Urgency keyword: '{word}'")
        for word in money_words:
            if word in text_lower:
                explanations.append(f"Money/prize keyword: '{word}'")
        for word in action_words:
            if word in text_lower:
                explanations.append(f"Action keyword: '{word}'")

        caps_ratio = sum(1 for c in text if c.isupper()) / len(text) if text else 0
        if caps_ratio > 0.3:
            explanations.append(f"High caps usage: {caps_ratio:.1%}")

        exclamation_count = text.count('!')
        if exclamation_count > 2:
            explanations.append(f"Multiple exclamation marks: {exclamation_count}")

        account_risk = 0
        if username:
            if any(word in username.lower() for word in ['winner', 'free', 'money', 'cash', 'prize']):
                account_risk += 3
                explanations.append(f"Suspicious username: '{username}'")
            if followers and followers < 50:
                account_risk += 1
                explanations.append(f"Low follower count: {followers}")

        text_risk = min(confidence * 100, 100) if pred == 'Fake' else 100 - (confidence * 100)
        overall_risk = min((text_risk + account_risk * 10) / 2, 100)

        return jsonify({
            'label': str(pred),
            'confidence': round(confidence, 3),
            'textRiskPct': round(text_risk, 1),
            'accountRiskPct': round(account_risk * 10, 1),
            'overallRiskPct': round(overall_risk, 1),
            'explanations': explanations[:5],
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'used_model': True,
            'metadata': {
                'username': username,
                'followers': followers,
                'text_length': len(text)
            }
        })
    except Exception as ex:
        print("Webhook error:", str(ex))
        return jsonify({'status': 'error', 'message': str(ex)}), 500


@app.route('/test-error', methods=['POST'])
def test_error():
    try:
        raise ValueError("Simulated error")
    except Exception as ex:
        print("Webhook error:", str(ex))
        return jsonify({'status': 'error', 'message': str(ex)}), 200


@app.route('/api/generate_report', methods=['POST'])
def api_generate_report():
    model, vectorizer = load_model()
    df = pd.read_csv(DATA_PATH)
    df['text'] = df['text'].apply(preprocess_text)

    X_tfidf = vectorizer.transform(df['text'])
    preds = model.predict(X_tfidf)
    acc = accuracy_score(df['label'], preds)
    report = classification_report(df['label'], preds, output_dict=True)

    metrics = {
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'accuracy': round(acc, 4),
        'precision': round(report['weighted avg']['precision'], 4),
        'recall': round(report['weighted avg']['recall'], 4),
        'f1': round(report['weighted avg']['f1-score'], 4)
    }
    return jsonify(metrics)

@app.route('/api/train', methods=['POST'])
def api_train():
    """Retrain the model on current dataset and return metrics."""
    try:
        model, vectorizer, acc = train_model()
        return jsonify({'status': 'ok', 'accuracy': round(float(acc), 4)})
    except Exception as ex:
        return jsonify({'status': 'error', 'message': str(ex)}), 500

# --------------------- Main ------------------------
if __name__ == '__main__':
    if not MODEL_PATH.exists():
        print("⚙️ No model found — training one now...")
        train_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

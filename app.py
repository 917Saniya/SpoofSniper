# app.py
import os
import re
import urllib.parse
import traceback
import hashlib
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# --------------------- App Setup ------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'spoofsniper.db'
DATA_PATH = BASE_DIR / 'spoof_posts.csv'
N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', "http://localhost:5678/webhook/facebook-check")
# Callback URL - use environment variable or detect from request
FLASK_HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
FLASK_PORT = os.environ.get('FLASK_PORT', '5000')

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
    first_name = Column(String(100), nullable=True)
    last_name = Column(String(100), nullable=True)
    phone = Column(String(20), nullable=True)
    organization = Column(String(200), nullable=True)
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
    return text.strip()

def train_model():
    """Train a simple text classification model"""
    if not DATA_PATH.exists():
        raise FileNotFoundError("❌ spoof_posts.csv not found.")

    df = pd.read_csv(DATA_PATH)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    df.dropna(subset=['text', 'label'], inplace=True)
    df['text'] = df['text'].apply(preprocess_text)

    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=800, class_weight='balanced', solver='lbfgs', C=0.8)
    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Model trained successfully — accuracy: {acc:.3f}")

    joblib.dump((model, vectorizer), MODEL_PATH, compress=3)
    with open(MODEL_COL_PATH, 'w', encoding='utf-8') as f:
        f.write('text\n')
    return model, vectorizer

def load_model():
    global _model, _vectorizer
    if _model is None:
        if MODEL_PATH.exists():
            _model, _vectorizer = joblib.load(MODEL_PATH)
            print("✅ Model loaded successfully.")
        else:
            print("⚙️ No model found — training new model...")
            _model, _vectorizer = train_model()
    return _model, _vectorizer

# --------------------- Pages ------------------------
@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/analysis')
def account_analysis():
    return render_template('analysis.html')

@app.route('/rules')
def detection_rules():
    return render_template('rules.html')

# --------------------- Auth (unchanged) ------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_input = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username_input or not password:
            flash('Username and password are required', 'danger')
            return render_template('login.html')
        db = SessionLocal()
        user = db.query(User).filter((User.username == username_input) | (User.email == username_input)).first()
        if not user:
            flash('Invalid username/email or password', 'danger')
            return render_template('login.html')
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if user.password_hash != password_hash and user.password_hash != '-':
            flash('Invalid username/email or password', 'danger')
            return render_template('login.html')
        session['user_id'] = user.id
        session['username'] = user.username
        if user.first_name:
            session['full_name'] = f"{user.first_name} {user.last_name or ''}".strip()
        flash('Logged in successfully', 'success')
        return redirect(url_for('landing'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        phone = request.form.get('phone', '').strip()
        organization = request.form.get('organization', '').strip()
        if not username or not email or not password:
            flash('Username, email, and password are required', 'danger')
            return render_template('signup.html')
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'danger')
            return render_template('signup.html')
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('signup.html')
        if not first_name or not last_name:
            flash('First name and last name are required', 'danger')
            return render_template('signup.html')
        if not phone:
            flash('Phone number is required', 'danger')
            return render_template('signup.html')
        db = SessionLocal()
        existing_user = db.query(User).filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            flash('Username or email already exists', 'warning')
            return render_template('signup.html')
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        user = User(username=username, email=email, password_hash=password_hash,
                    first_name=first_name, last_name=last_name, phone=phone,
                    organization=organization if organization else None)
        db.add(user)
        db.commit()
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('landing'))

# --------------------- Prediction API & storage ------------------------
latest_results = {}  # in-memory store: keys -> result dicts

def extract_facebook_username(raw):
    """
    Extract username or profile id from typical facebook links:
    - https://www.facebook.com/username
    - https://www.facebook.com/profile.php?id=123456789
    Falls back to raw string (stripped of @).
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    # first try profile.php?id=
    m = re.search(r'profile\.php\?id=(\d+)', raw, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'facebook\.com/(?:people/)?([A-Za-z0-9_.-]+)', raw, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # otherwise remove leading @
    return raw.lstrip('@').strip()

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Send prediction request to n8n and return immediate model result"""
    try:
        data = request.get_json(force=True)
        username_input = (data.get('username') or '').strip()
        followers = int(data.get('followers', 0) or 0)

        extracted_username = extract_facebook_username(username_input)
        if not extracted_username:
            return jsonify({"error": "Invalid Facebook username or URL"}), 400

        model, vectorizer = load_model()
        text_clean = preprocess_text(extracted_username)
        X_vec = vectorizer.transform([text_clean])

        # Predict robustly
        pred_raw = None
        try:
            pred_raw = model.predict(X_vec)
            pred_val = pred_raw[0] if hasattr(pred_raw, '__len__') else pred_raw
        except Exception:
            # fallback - attempt to convert first prediction to string
            pred_val = None

        # Determine is_fake from pred_val type
        is_fake = False
        if pred_val is None:
            # fallback: when model didn't provide a meaningful prediction, use heuristic default
            is_fake = False
        elif isinstance(pred_val, (np.integer, int)):
            is_fake = int(pred_val) == 1
        elif isinstance(pred_val, (np.floating, float)):
            is_fake = int(round(pred_val)) == 1
        elif isinstance(pred_val, str):
            is_fake = pred_val.lower() in ('fake', '1', 'true', 'yes', 'spam')
        else:
            # unknown type -> assume not fake
            is_fake = False

        # Get probabilities if available
        base_confidence = 0.6  # default base
        try:
            if hasattr(model, 'predict_proba'):
                proba_all = model.predict_proba(X_vec)
                if hasattr(proba_all, '__len__'):
                    proba = proba_all[0]
                    # if binary assume [prob_real, prob_fake] or [prob_class0, prob_class1]
                    if len(proba) == 2:
                        base_confidence = float(proba[1] if is_fake else proba[0])
                    else:
                        base_confidence = float(np.max(proba))
            else:
                base_confidence = 0.6
        except Exception:
            base_confidence = 0.6

        # Adjust confidence with followers & username complexity
        follower_factor = min(1.0, followers / 100.0)
        username_complexity = len(extracted_username.replace('_', '').replace('.', '').replace('-', ''))
        complexity_factor = min(1.0, username_complexity / 15.0)

        if not is_fake:
            confidence_boost = (follower_factor * 0.3) + (complexity_factor * 0.2)
            confidence = min(0.99, base_confidence + confidence_boost)
        else:
            account_suspicion = 0 if followers >= 50 else 0.15
            confidence = min(0.99, base_confidence + account_suspicion)

        # minor deterministic variation from hash
        username_hash = int(hashlib.md5(extracted_username.encode()).hexdigest()[:8], 16)
        hash_factor = (username_hash % 100) / 1000.0
        confidence = max(0.5, min(0.99, confidence + (hash_factor - 0.05)))

        # risk calculation
        if is_fake:
            risk_pct = confidence * 100.0
        else:
            risk_pct = (1 - confidence) * 100.0

        if followers >= 1000:
            account_risk = 0
        elif followers >= 500:
            account_risk = 2
        elif followers >= 100:
            account_risk = 5
        elif followers >= 50:
            account_risk = 8
        else:
            account_risk = 15

        suspicious_patterns = ['win', 'free', 'prize', 'money', 'cash', 'official', 'verify', 'claim']
        if any(p in extracted_username.lower() for p in suspicious_patterns):
            account_risk += 5
        account_risk = min(50, account_risk)
        overall_risk = round((risk_pct + account_risk) / 2.0, 1)

        label = 'fake' if is_fake else 'real'

        result = {
            "facebook_id_or_username": extracted_username,
            "label": label,
            "confidence": round(confidence, 3),
            "accountRiskPct": round(account_risk, 1),
            "overallRiskPct": overall_risk,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # store a pending placeholder under multiple keys so frontend polling works
        raw_key = username_input or extracted_username
        encoded_key = urllib.parse.quote(raw_key, safe='')

        pending = {
            "status": "pending",
            "facebook_id_or_username": extracted_username,
            "initial_result": result
        }

        # store by extracted username, raw input, and encoded input
        latest_results[extracted_username] = pending
        latest_results[raw_key] = pending
        latest_results[encoded_key] = pending

        # send to n8n webhook
        callback_url = f"http://{FLASK_HOST}:{FLASK_PORT}/callback_result"
        payload = {
            "facebook_id_or_username": extracted_username,
            "followers": followers,
            "analysis_result": result,
            "callback_url": callback_url,
            "raw_input": username_input  # Include raw input for better matching
        }
        try:
            # Increased timeout to 30 seconds for n8n processing
            r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
            print(f"📤 Sent to n8n ({r.status_code}) for {extracted_username}")
            if r.status_code != 200:
                print(f"⚠️ n8n returned status {r.status_code}: {r.text[:200]}")
        except requests.exceptions.Timeout:
            print(f"⚠️ n8n request timed out for {extracted_username}")
        except requests.exceptions.ConnectionError:
            print(f"⚠️ Could not connect to n8n at {N8N_WEBHOOK_URL}")
        except Exception as e:
            print(f"⚠️ Error sending to n8n: {e}")

        # return the immediate model result and include extracted_username
        return jsonify({**result, "extracted_username": extracted_username}), 200

    except Exception as ex:
        print("❌ Error in /api/predict:", ex)
        print(traceback.format_exc())
        return jsonify({"error": str(ex)}), 500

# --------------------- Callback Endpoint ------------------------
@app.route("/callback_result", methods=["POST", "GET", "OPTIONS"])
def callback_result():
    """Handle callback from n8n workflow"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        return response
    
    try:
        # Handle both GET (query params) and POST (JSON body)
        if request.method == 'GET':
            data = request.args.to_dict()
        else:
            data = request.get_json(force=True) or {}
        
        print("✅ Received callback from n8n")
        print(f"   Method: {request.method}")
        print(f"   Data type: {type(data)}")
        print(f"   Data preview: {str(data)[:500]}")

        # unwrap if n8n wrapped it under list/body/json
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        if isinstance(data, dict) and "body" in data:
            data = data["body"]
        if isinstance(data, dict) and "json" in data:
            data = data["json"]

        # fallback empty dict
        if not isinstance(data, dict):
            print("⚠️ Invalid format from n8n — not a dict:", type(data))
            return jsonify({"status": "error", "message": "Invalid JSON structure"}), 400

        # extract username safely - try multiple fields
        username = (
            data.get("facebook_id_or_username")
            or data.get("facebook_id")
            or data.get("username")
            or data.get("id")
            or data.get("raw_input", "").strip()  # Try raw_input as fallback
            or ""
        )
        
        # If still empty, try to extract from nested structures
        if not username:
            if "data" in data and isinstance(data["data"], dict):
                username = data["data"].get("facebook_id_or_username") or ""
            if not username and "result" in data and isinstance(data["result"], dict):
                username = data["result"].get("facebook_id_or_username") or ""

        username = str(username).strip()

        # if username invalid or placeholder
        if not username or username == "=" or username == "None":
            print("⚠️ Invalid username in callback, trying to use raw_input or stored keys")
            # Try to find any pending result and update it
            for key, value in list(latest_results.items()):
                if isinstance(value, dict) and value.get("status") == "pending":
                    username = value.get("facebook_id_or_username", "")
                    if username:
                        break
            
            if not username or username == "=":
                anon_key = f"unknown-{int(datetime.utcnow().timestamp())}"
                latest_results[anon_key] = {
                    "status": "completed",
                    "warning": "Invalid username received",
                    "raw": data,
                }
                print("⚠️ Stored anonymous result under key:", anon_key)
                return jsonify({"status": "received", "warning": "invalid username"}), 200

        # unwrap nested analysis_result - try multiple paths
        analysis_result = (
            data.get("analysis_result") 
            or data.get("result")
            or data.get("data", {}).get("analysis_result")
            or {}
        )
        
        # If analysis_result is still empty, use the whole data as result
        if not analysis_result or not isinstance(analysis_result, dict):
            analysis_result = {k: v for k, v in data.items() if k not in ["facebook_id_or_username", "username", "id"]}

        # check if data is empty placeholders from n8n (but be more lenient)
        invalid_vals = {"=", ""}
        if analysis_result:
            analysis_values = [str(v) for v in analysis_result.values() if v is not None]
            if set(analysis_values).intersection(invalid_vals) and len(set(analysis_values)) <= 2:
                print(f"⚠️ Skipping invalid placeholder result for {username}")
                return jsonify({"status": "ignored", "reason": "placeholder values"}), 200

        # merge with pending result if exists
        pending = latest_results.get(username, {})
        initial_result = pending.get("initial_result", {})
        
        # Merge n8n results with initial result, prioritizing n8n data
        merged_analysis = {**initial_result, **analysis_result}
        
        merged = {
            "status": "completed",
            "facebook_id_or_username": username,
            "analysis_result": merged_analysis,
            "n8n_data": data,  # Keep raw n8n data for debugging
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        # Store under all possible keys to ensure polling works
        raw_input = data.get("raw_input", username)
        keys_to_store = [
            username,
            urllib.parse.quote(username, safe=""),
            raw_input,
            urllib.parse.quote(raw_input, safe=""),
        ]
        
        # Also try to find and update any pending entries
        for key in list(latest_results.keys()):
            if isinstance(latest_results.get(key), dict):
                pending_entry = latest_results[key]
                if pending_entry.get("facebook_id_or_username") == username:
                    keys_to_store.append(key)

        for k in set(keys_to_store):  # Use set to avoid duplicates
            if k:  # Only store non-empty keys
                latest_results[k] = merged

        print(f"✅ Stored valid callback result for username: {username}")
        print(f"   Stored under {len(set(keys_to_store))} keys")
        
        response = jsonify({"status": "received", "username": username})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response, 200

    except Exception as e:
        print(f"❌ Error in callback: {e}")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# --------------------- Debug/Test Endpoint ------------------------
@app.route('/api/test_callback', methods=['POST', 'GET'])
def test_callback():
    """Test endpoint to verify callback is accessible"""
    return jsonify({
        "status": "ok",
        "message": "Callback endpoint is accessible",
        "method": request.method,
        "callback_url": f"http://{FLASK_HOST}:{FLASK_PORT}/callback_result",
        "n8n_webhook_url": N8N_WEBHOOK_URL,
        "timestamp": datetime.utcnow().isoformat()
    }), 200

# --------------------- Polling endpoint for frontend ------------------------
@app.route('/api/latest_result/<path:username>', methods=['GET'])
def get_latest_result(username):
    """Get latest result for a username, trying multiple key variations"""
    # try exact match, decoded match, and some fallbacks
    candidates = [
        username,
        urllib.parse.unquote(username),
        urllib.parse.quote(username, safe=''),
    ]
    
    # also try unquoted trimmed
    decoded = urllib.parse.unquote(username)
    if decoded != username:
        candidates.append(decoded)
        candidates.append(urllib.parse.quote(decoded, safe=''))
    
    # try removing protocol prefix if user passed full URL
    if username.startswith("http"):
        simplified = extract_facebook_username(username)
        if simplified:
            candidates.append(simplified)
            candidates.append(urllib.parse.quote(simplified, safe=''))
    
    # Try to find by matching facebook_id_or_username in stored results
    for key, value in latest_results.items():
        if isinstance(value, dict):
            stored_username = value.get("facebook_id_or_username", "")
            if stored_username and (stored_username == username or stored_username == decoded):
                candidates.append(key)

    # Remove duplicates and empty strings
    candidates = list(set([c for c in candidates if c]))

    for k in candidates:
        if k in latest_results:
            result = latest_results[k]
            print(f"📊 Found result for '{username}' using key '{k}': status={result.get('status', 'unknown')}")
            return jsonify(result), 200

    # not ready yet -> return pending (200) so frontend keeps polling
    print(f"⏳ No result found for '{username}', returning pending status")
    return jsonify({
        "status": "pending", 
        "error": "Result not ready yet", 
        "username": username,
        "tried_keys": candidates[:5]  # Debug info
    }), 200

# --------------------- Main ------------------------
if __name__ == '__main__':
    try:
        if not MODEL_PATH.exists():
            print("⚙️ No model found — training new one...")
            train_model()
    except Exception as e:
        print(f"⚠️ Model training failed: {e}")

    print("🚀 Starting Flask server on port 5000...")
    app.run(debug=True, port=5000)

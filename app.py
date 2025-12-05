import os
import re
import hashlib
import urllib.parse
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_cors import CORS
import requests

# --------------------- App Setup ------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
CORS(app)

N8N_WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', "http://localhost:5678/webhook/facebook-check")
FLASK_HOST = os.environ.get('FLASK_HOST', '127.0.0.1')
FLASK_PORT = int(os.environ.get('FLASK_PORT', 5000))

latest_results = {}  # in-memory store for results
users = {}  # simple in-memory user store: {username: {password_hash, first_name, last_name}}

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

# --------------------- Auth ------------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username_input = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username_input or not password:
            flash('Username and password are required', 'danger')
            return render_template('login.html')
        user = users.get(username_input)
        if not user or user['password_hash'] != hashlib.sha256(password.encode()).hexdigest():
            flash('Invalid username or password', 'danger')
            return render_template('login.html')
        session['username'] = username_input
        session['full_name'] = f"{user.get('first_name','')} {user.get('last_name','')}".strip()
        flash('Logged in successfully', 'success')
        return redirect(url_for('landing'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        if not username or not password or not first_name or not last_name:
            flash('All fields are required', 'danger')
            return render_template('signup.html')
        if password != confirm_password:
            flash('Passwords do not match', 'danger')
            return render_template('signup.html')
        if username in users:
            flash('Username already exists', 'warning')
            return render_template('signup.html')
        users[username] = {
            'password_hash': hashlib.sha256(password.encode()).hexdigest(),
            'first_name': first_name,
            'last_name': last_name
        }
        flash('Account created successfully! Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('landing'))

# --------------------- Helper functions ------------------------
def extract_facebook_username(raw):
    raw = (raw or "").strip()
    if not raw:
        return ""
    m = re.search(r'profile\.php\?id=(\d+)', raw, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r'facebook\.com/(?:people/)?([A-Za-z0-9_.-]+)', raw, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return raw.lstrip('@').strip()

def predict_username_fake(username, followers):
    username_clean = username.lower()
    score = 0.0

    # Suspicious words
    suspicious_words = ['win', 'free', 'prize', 'money', 'cash', 'official', 'verify', 'claim']
    for word in suspicious_words:
        if word in username_clean:
            score += 0.3

    # Username length
    if len(username_clean) < 5:
        score += 0.2
    elif len(username_clean) > 20:
        score += 0.1

    # Special characters
    specials = sum([1 for c in username_clean if not c.isalnum()])
    score += min(0.2, specials * 0.05)

    # Numbers
    numbers = sum([1 for c in username_clean if c.isdigit()])
    score += min(0.2, numbers * 0.05)

    # Repeated characters
    repeated_chars = max([username_clean.count(c) for c in set(username_clean)])
    if repeated_chars > 3:
        score += 0.1

    # Followers impact
    if followers < 20:
        score += 0.3
    elif followers < 50:
        score += 0.2
    elif followers < 100:
        score += 0.1

    # Cap score
    score = min(0.99, score)

    # Label and confidence
    label = 'fake' if score > 0.4 else 'real'
    confidence = max(0.5, score) if label == 'fake' else max(0.5, 1 - score)
    return label, round(confidence, 3)

# --------------------- Prediction API ------------------------
@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        username_input = (data.get('username') or '').strip()
        followers = int(data.get('followers', 0) or 0)
        extracted_username = extract_facebook_username(username_input)
        if not extracted_username:
            return jsonify({"error": "Invalid Facebook username or URL"}), 400

        label, confidence = predict_username_fake(extracted_username, followers)

        # Account risk calculation
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

        suspicious_words = ['win', 'free', 'prize', 'money', 'cash', 'official', 'verify', 'claim']
        if any(p in extracted_username.lower() for p in suspicious_words):
            account_risk += 5
        account_risk = min(50, account_risk)
        overall_risk = round((confidence*100 + account_risk)/2.0, 1)

        result = {
            "facebook_id_or_username": extracted_username,
            "label": label,
            "confidence": confidence,
            "accountRiskPct": account_risk,
            "overallRiskPct": overall_risk,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        raw_key = username_input or extracted_username
        encoded_key = urllib.parse.quote(raw_key, safe='')
        latest_results[extracted_username] = result
        latest_results[raw_key] = result
        latest_results[encoded_key] = result

        # Send to n8n webhook
        payload = {
            "facebook_id_or_username": extracted_username,
            "followers": followers,
            "analysis_result": result
        }
        try:
            r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=30)
            print(f"📤 Sent to n8n ({r.status_code}) for {extracted_username}")
        except Exception as e:
            print(f"⚠️ Could not send to n8n: {e}")

        return jsonify(result), 200
    except Exception as ex:
        print("❌ Error in /api/predict:", ex)
        return jsonify({"error": str(ex)}), 500

# --------------------- Callback Endpoint ------------------------
@app.route("/callback_result", methods=["POST", "GET", "OPTIONS"])
def callback_result():
    if request.method == 'OPTIONS':
        response = jsonify({"status": "ok"})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        return response
    try:
        if request.method == 'GET':
            data = request.args.to_dict()
        else:
            data = request.get_json(force=True) or {}
        username = (
            data.get("facebook_id_or_username") or
            data.get("username") or
            data.get("id") or
            data.get("raw_input", "").strip() or ""
        )
        username = str(username).strip()
        if not username:
            return jsonify({"status": "error", "message": "Invalid username"}), 400
        analysis_result = data.get("analysis_result") or {}
        latest_results[username] = analysis_result
        latest_results[urllib.parse.quote(username, safe='')] = analysis_result
        print(f"✅ Stored callback result for {username}")
        return jsonify({"status": "received", "username": username}), 200
    except Exception as e:
        print(f"❌ Error in callback: {e}")
        return jsonify({"error": str(e)}), 500

# --------------------- Polling endpoint ------------------------
@app.route('/api/latest_result/<path:username>', methods=['GET'])
def get_latest_result(username):
    username_clean = urllib.parse.unquote(username)
    for key in [username, username_clean, urllib.parse.quote(username_clean, safe='')]:
        if key in latest_results:
            return jsonify(latest_results[key]), 200
    return jsonify({"status": "pending", "username": username}), 200

# --------------------- Test endpoint ------------------------
@app.route('/api/test_callback', methods=['GET', 'POST'])
def test_callback():
    return jsonify({
        "status": "ok",
        "message": "Callback endpoint is accessible",
        "callback_url": f"http://{FLASK_HOST}:{FLASK_PORT}/callback_result",
        "n8n_webhook_url": N8N_WEBHOOK_URL,
        "timestamp": datetime.utcnow().isoformat()
    }), 200

# --------------------- Main ------------------------
if __name__ == '__main__':
    print(f"🚀 Starting Flask server on {FLASK_HOST}:{FLASK_PORT}...")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=True)

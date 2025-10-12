from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///spoofsniper.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Database Models
class Post(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    username = db.Column(db.String(100))
    followers = db.Column(db.Integer)
    account_age_days = db.Column(db.Integer)
    prediction = db.Column(db.String(10))  # 'fake' or 'real'
    confidence = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_feedback = db.Column(db.String(10))  # 'correct' or 'incorrect'

class ModelMetrics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    accuracy = db.Column(db.Float)
    precision = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1_score = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize ML components
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fake post indicators
FAKE_INDICATORS = [
    r'\b(urgent|emergency|act now|limited time|click here|free money|winner|congratulations)\b',
    r'\b(verify|confirm|update|suspended|locked|hacked)\b',
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    r'\$\d+|\d+\$',  # Money amounts
    r'[!]{2,}',  # Multiple exclamation marks
]

def extract_features(text, username=None, followers=None, account_age=None):
    """Extract features from post text and metadata"""
    features = {}
    
    # Text-based features
    text_lower = text.lower()
    features['length'] = len(text)
    features['word_count'] = len(text.split())
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0
    
    # Fake indicators
    fake_score = 0
    for pattern in FAKE_INDICATORS:
        matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
        fake_score += matches
    
    features['fake_indicators'] = fake_score
    
    # Metadata features
    if followers is not None:
        features['followers'] = followers
        features['low_followers'] = 1 if followers < 100 else 0
    else:
        features['followers'] = 0
        features['low_followers'] = 0
    
    if account_age is not None:
        features['account_age'] = account_age
        features['new_account'] = 1 if account_age < 30 else 0
    else:
        features['account_age'] = 0
        features['new_account'] = 0
    
    return features

def train_model():
    """Train the ML model with sample data"""
    # Create sample training data
    sample_data = [
        # Fake posts
        ("URGENT! Click here to claim your $1000 prize! Limited time offer!", "fake"),
        ("Congratulations! You've won $500! Verify your account now!", "fake"),
        ("Your account will be suspended! Click link to verify immediately!", "fake"),
        ("FREE MONEY! Act now! Limited time offer! Click here!", "fake"),
        ("Emergency! Your account is hacked! Verify immediately!", "fake"),
        ("Winner! You've been selected for $1000! Click to claim!", "fake"),
        ("URGENT UPDATE: Verify your account or it will be locked!", "fake"),
        ("Congratulations! You won $500! Click here to claim!", "fake"),
        
        # Real posts
        ("Just had a great day at the park with my family", "real"),
        ("Looking forward to the weekend! Anyone have fun plans?", "real"),
        ("Thanks everyone for the birthday wishes!", "real"),
        ("Beautiful sunset today. Nature never fails to amaze me", "real"),
        ("Coffee and a good book - perfect morning", "real"),
        ("Excited about the new project I'm working on", "real"),
        ("Had an amazing dinner at the new restaurant downtown", "real"),
        ("Weekend vibes! Time to relax and recharge", "real"),
    ]
    
    texts = [item[0] for item in sample_data]
    labels = [item[1] for item in sample_data]
    
    # Extract features
    features_list = []
    for text in texts:
        features = extract_features(text)
        features_list.append(features)
    
    # Convert to DataFrame
    df = pd.DataFrame(features_list)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, labels, test_size=0.2, random_state=42
    )
    
    # Train classifier
    classifier.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Save metrics to database
    metrics = ModelMetrics(
        accuracy=accuracy,
        precision=0.85,  # Placeholder values
        recall=0.80,
        f1_score=0.82
    )
    db.session.add(metrics)
    db.session.commit()
    
    return accuracy

def predict_post(text, username=None, followers=None, account_age=None):
    """Predict if a post is fake or real"""
    features = extract_features(text, username, followers, account_age)
    features_df = pd.DataFrame([features])
    
    prediction = classifier.predict(features_df)[0]
    confidence = max(classifier.predict_proba(features_df)[0])
    
    return prediction, confidence, features

def get_explanation(features, prediction):
    """Generate explanation for the prediction"""
    explanations = []
    
    if features['fake_indicators'] > 0:
        explanations.append(f"Contains {features['fake_indicators']} suspicious keywords/phrases")
    
    if features['caps_ratio'] > 0.3:
        explanations.append("High use of capital letters (often used in fake posts)")
    
    if features['exclamation_count'] > 2:
        explanations.append("Multiple exclamation marks (common in fake posts)")
    
    if features['low_followers']:
        explanations.append("Low follower count (potential red flag)")
    
    if features['new_account']:
        explanations.append("New account (less than 30 days old)")
    
    if features['length'] < 50:
        explanations.append("Very short post (unusual for genuine content)")
    
    if not explanations:
        explanations.append("No obvious red flags detected")
    
    return explanations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        # Add authentication logic here
        flash('Login functionality coming soon!', 'info')
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        email = request.form.get('email')
        password = request.form.get('password')
        # Add registration logic here
        flash('Registration functionality coming soon!', 'info')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/detection_rules')
def detection_rules():
    return render_template('detection_rules.html')

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        username = request.form.get('username', '').strip()
        followers = request.form.get('followers', type=int)
        account_age = request.form.get('account_age', type=int)
        
        if not text:
            flash('Please enter some text to analyze', 'error')
            return redirect(url_for('analyze'))
        
        # Make prediction
        prediction, confidence, features = predict_post(text, username, followers, account_age)
        explanations = get_explanation(features, prediction)
        
        # Save to database
        post = Post(
            text=text,
            username=username,
            followers=followers,
            account_age_days=account_age,
            prediction=prediction,
            confidence=confidence
        )
        db.session.add(post)
        db.session.commit()
        
        return render_template('results.html', 
                             text=text,
                             prediction=prediction,
                             confidence=confidence,
                             explanations=explanations,
                             features=features,
                             post_id=post.id)
    
    return render_template('analyze.html')

@app.route('/feedback', methods=['POST'])
def feedback():
    post_id = request.form.get('post_id')
    feedback = request.form.get('feedback')
    
    if post_id and feedback:
        post = Post.query.get(post_id)
        if post:
            post.user_feedback = feedback
            db.session.commit()
            flash('Thank you for your feedback!', 'success')
    
    return redirect(url_for('index'))

@app.route('/admin')
def admin():
    # Get statistics
    total_posts = Post.query.count()
    fake_posts = Post.query.filter_by(prediction='fake').count()
    real_posts = Post.query.filter_by(prediction='real').count()
    
    # Get recent posts
    recent_posts = Post.query.order_by(Post.created_at.desc()).limit(10).all()
    
    # Get model metrics
    latest_metrics = ModelMetrics.query.order_by(ModelMetrics.created_at.desc()).first()
    
    return render_template('admin.html',
                         total_posts=total_posts,
                         fake_posts=fake_posts,
                         real_posts=real_posts,
                         recent_posts=recent_posts,
                         metrics=latest_metrics)

@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    stats = {
        'total_posts': Post.query.count(),
        'fake_posts': Post.query.filter_by(prediction='fake').count(),
        'real_posts': Post.query.filter_by(prediction='real').count(),
        'avg_confidence': db.session.query(db.func.avg(Post.confidence)).scalar() or 0
    }
    return jsonify(stats)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Train the model on startup
        train_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)

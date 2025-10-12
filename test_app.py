#!/usr/bin/env python3
"""
Test script for SpoofSniper
This script tests the core functionality without running the web server
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import extract_features, predict_post, get_explanation

def test_fake_post():
    """Test detection of a fake post"""
    print("Testing FAKE post detection...")
    fake_text = "URGENT! Click here to claim your $1000 prize! Limited time offer!"
    
    prediction, confidence, features = predict_post(fake_text)
    explanations = get_explanation(features, prediction)
    
    print(f"Text: {fake_text}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Features: {features}")
    print(f"Explanations: {explanations}")
    print("-" * 50)

def test_real_post():
    """Test detection of a real post"""
    print("Testing REAL post detection...")
    real_text = "Just had a great day at the park with my family"
    
    prediction, confidence, features = predict_post(real_text)
    explanations = get_explanation(features, prediction)
    
    print(f"Text: {real_text}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2%}")
    print(f"Features: {features}")
    print(f"Explanations: {explanations}")
    print("-" * 50)

def test_feature_extraction():
    """Test feature extraction function"""
    print("Testing feature extraction...")
    text = "URGENT! Click here to claim your $1000 prize!"
    features = extract_features(text, username="test_user", followers=50, account_age=5)
    
    print(f"Text: {text}")
    print(f"Extracted features: {features}")
    print("-" * 50)

if __name__ == "__main__":
    print("SpoofSniper Test Suite")
    print("=" * 50)
    
    try:
        test_feature_extraction()
        test_fake_post()
        test_real_post()
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


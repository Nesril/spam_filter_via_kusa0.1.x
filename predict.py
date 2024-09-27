# predict.py

import joblib
import sys
from fetch_data import RemoteDataset
from kusa import DatasetClient, DatasetSDKException
import pandas as pd
import os
from dotenv import load_dotenv


def lemmatize_text(text, nlp, stop_words):
    import re
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    lemma_text = []
    doc = nlp(text)
    for token in doc:
        if token.lemma_ not in stop_words and not token.is_punct and not token.like_num:
            lemma_text.append(token.lemma_)
    return " ".join(lemma_text)

def main():
    if len(sys.argv) < 2:
        print("Please provide at least one email to classify.")
        return
    
    emails = sys.argv[1:]
    
    # Load the saved model and vectorizer
    model = joblib.load('spam_classifier.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    # Load SpaCy model
    import spacy
    nlp = spacy.load("en_core_web_sm")
    stop_words = nlp.Defaults.stop_words
    
    # Preprocess emails
    lemmatize = lambda text: lemmatize_text(text, nlp, stop_words)
    checked_emails = [lemmatize(email) for email in emails]
    
    # Vectorize
    emails_count = vectorizer.transform(checked_emails)
    
    # Predict
    predictions = model.predict(emails_count)
    
    # Display predictions
    for email, prediction in zip(emails, predictions):
        print(f"Email: {email} => Prediction: {'spam' if prediction == 1 else 'ham'}")

if __name__ == "__main__":
    main()

import tensorflow as tf
import pickle
import spacy
import re
from dotenv import load_dotenv
from kusa import DatasetClient
import os

def lemmatize_text(text, nlp, stop_words):
    """Secure text preprocessing matching training pipeline"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    lemma_text = []
    doc = nlp(text)
    for token in doc:
        if (token.lemma_ not in stop_words and 
            not token.is_punct and 
            not token.like_num):
            lemma_text.append(token.lemma_)
    return " ".join(lemma_text)

def load_artifacts():
    """Load model and tokenizer securely"""
    try:
        # Load Keras model
        model = tf.keras.models.load_model('secure_spam_model.keras')
        
        # Load tokenizer
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except Exception as e:
        raise ValueError(f"Error loading model artifacts: {str(e)}")

def predict_emails(emails, model, tokenizer, nlp, stop_words):
    """Secure prediction pipeline"""
    # Preprocess
    processed_emails = [
        lemmatize_text(email, nlp, stop_words) 
        for email in emails
    ]
    
    # Tokenize and pad
    sequences = tokenizer.texts_to_sequences(processed_emails)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=100,
        padding='post'
    )
    
    # Predict
    predictions = model.predict(padded)
    return (predictions > 0.5).astype("int32").flatten()

def main():
    # Sample emails - replace with sys.argv in production
    emails = [
        'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!',
        'Hey mohan, can we get together to watch footbal game tomorrow?',
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    ]
    
    # Load NLP model
    nlp = spacy.load("en_core_web_sm")
    stop_words = nlp.Defaults.stop_words
    
    try:
        # Load model artifacts
        model, tokenizer = load_artifacts()
        
        # Get predictions
        predictions = predict_emails(emails, model, tokenizer, nlp, stop_words)
        
        # Display results
        print("\nEmail Classification Results:")
        for email, pred in zip(emails, predictions):
            print(f"✉️ {email[:60]}... => {'🚨 SPAM' if pred == 1 else '📨 HAM'}")
            
        print("\nPrediction confidence:")
        probs = model.predict(
            tf.keras.preprocessing.sequence.pad_sequences(
                tokenizer.texts_to_sequences([
                    lemmatize_text(e, nlp, stop_words) for e in emails
                ]),
                maxlen=100,
                padding='post'
            )
        )
        for email, prob in zip(emails, probs):
            print(f"{'SPAM' if prob > 0.5 else 'HAM'} ({prob[0]:.2%}): {email[:45]}...")
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    load_dotenv()
    main()
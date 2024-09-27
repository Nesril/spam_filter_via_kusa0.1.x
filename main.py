import os
import pandas as pd
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from kusa import DatasetClient, DatasetSDKException
from fetch_data import RemoteDataset
import spacy

load_dotenv()

# Configuration
PUBLIC_ID = os.getenv('PUBLIC_ID')
SECRET_KEY = os.getenv('SECRET_KEY')
BATCH_SIZE = 100

# Load SpaCy model for text processing
nlp = spacy.load("en_core_web_sm")
stop_words = nlp.Defaults.stop_words

def Lemmatize_text(text):
    lemma_text = []
    doc = nlp(text)
    for token in doc:
        if token.lemma_ not in stop_words and not token.is_punct:
            lemma_text.append(token.lemma_)
    return " ".join(lemma_text)

def main():
    # Initialize the SDK client
    client = DatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)

    try:
        init_data = client.initialize()
        print(f"Total Rows: {init_data['totalRows']}")
        print("First 10 Rows:")
        print(init_data['first10Rows'])
    except DatasetSDKException as e:
        print(f"Initialization error: {e}")
        return

    # Create the dataset
    dataset = RemoteDataset(client=client, batch_size=BATCH_SIZE)
    
    # Fetch all data for processing
    inputs, labels = dataset.get_all_data()  # Implement this method in your RemoteDataset
    print("Fetched inputs and labels.")

    labels = pd.DataFrame(labels, columns=['Category'])
    inputs = pd.DataFrame(inputs, columns=['Message'])

    merged_df = pd.concat([labels, inputs], axis=1)
    
    print("merged_df ",merged_df)
    x=merged_df["Category"]
    x=x.apply(lambda s:1 if s=="spam" else 0)
    y=merged_df["Message"]
    y=y.apply(Lemmatize_text)

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(y, x, test_size=0.25, random_state=0)

    # Vectorize the text data using TF-IDF
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(x_train)
    X_test_vectorized = vectorizer.transform(x_test)

    # Train the Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train_vectorized, y_train)

    # Evaluate the model
    accuracy = model.score(X_test_vectorized, y_test)
    print(f"Accuracy on the test set: {accuracy:.2f}")

    # Example emails to check predictions
    emails = [
        'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!',
        'Hey mohan, can we get together to watch footbal game tomorrow?',
        "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"
    ]

    # Process and predict on new emails
    checked_emails = [Lemmatize_text(email) for email in emails]
    emails_count = vectorizer.transform(checked_emails)
    predictions = model.predict(emails_count)

    # Display predictions
    for email, prediction in zip(checked_emails, predictions):
        print(f"Email: {email} => Prediction: {'spam' if prediction == 1 else 'ham'}")

if __name__ == "__main__":
    main()

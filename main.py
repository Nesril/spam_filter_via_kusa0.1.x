import os
from dotenv import load_dotenv
from kusa.client import SecureDatasetClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

load_dotenv()

def train_model(X, y, **params):
    model = LogisticRegression(
        **params,
        class_weight='balanced',
    )
    model.fit(X, y)
    return model

def main():
    # Load credentials
    PUBLIC_ID = os.getenv("PUBLIC_ID")
    SECRET_KEY = os.getenv("SECRET_KEY")

    # Step 1: Initialize secure client
    client = SecureDatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
    initialization = client.initialize()
    # Step 2: Load encrypted dataset into memory
    client.fetch_and_decrypt_batch(batch_size=500, batch_number=1)

    # Step 3: Configure preprocessing
    client.configure_preprocessing({
         "tokenizer": "nltk",
    "stopwords": True,
    "reduction": "tfidf",
    "target_column": "Category"
    })
    client.run_preprocessing()

    # Step 4: Train model using internal data
    client.train(
        user_train_func=train_model,
        hyperparams={"max_iter": 1000},
        target_column="Category"  # Make sure this column is your label (e.g., spam/ham)
    )

    # Step 5: Evaluate the model
    results = client.evaluate()
    print("\n✅ Evaluation Accuracy:", results["accuracy"])
    print("📊 Classification Report:\n", results["report"])

    # Step 6: Visualize Confusion Matrix
    y_true = client._SecureDatasetClient__y_val
    y_pred = client._SecureDatasetClient__trained_model.predict(client._SecureDatasetClient__X_val)
    cm = confusion_matrix(y_true, y_pred)

    print("y_true ",y_true)
    print("y_pred ",y_pred)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # Step 7: Save the trained model
    client.save_model("secure_spam_model.joblib")

   

if __name__ == "__main__":
    main()

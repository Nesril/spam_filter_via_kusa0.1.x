import os
import tensorflow as tf
from dotenv import load_dotenv
from kusa import DatasetClient
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from fetch_data import SecureSpamDataset
import pickle

# Load environment variables
load_dotenv()

def main():
    # Configuration
    PUBLIC_ID = os.getenv('PUBLIC_ID')
    SECRET_KEY = os.getenv('SECRET_KEY')
    BATCH_SIZE = 100
    EPOCHS = 5

    # Initialize secure client
    client = DatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)

    try:
        # Create secure dataset
        print("Initializing secure dataset...")
        dataset = SecureSpamDataset(client=client, batch_size=BATCH_SIZE)
        
        # Create and prepare TensorFlow datasets
        def prepare_datasets():
            full_dataset = dataset.get_tf_dataset()
            
            # Calculate dataset sizes
            total_batches = len(dataset) // BATCH_SIZE
            train_size = int(0.8 * total_batches)
            
            # Repeat dataset indefinitely for training
            train_ds = full_dataset.take(train_size).repeat()
            test_ds = full_dataset.skip(train_size)
            
            return train_ds, test_ds, train_size

        train_ds, test_ds, train_steps = prepare_datasets()
        
        # Build model
        print("Building model...")
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=len(dataset.tokenizer.word_index) + 1,
                output_dim=64,
                mask_zero=True
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        # Train model with proper steps_per_epoch
        print("Training model...")
        history = model.fit(
            train_ds,
            steps_per_epoch=train_steps,
            validation_data=test_ds,
            epochs=EPOCHS
        )
        
        # Evaluate
        print("\nEvaluation Results:")
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Generate predictions for classification report
        print("\nGenerating classification report...")
        y_true, y_pred = [], []
        for batch in test_ds:
            texts, labels = batch
            preds = (model.predict(texts, verbose=0) > 0.5).astype("int32")
            y_true.extend(labels.numpy())
            y_pred.extend(preds.flatten())
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['ham', 'spam']))
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training History')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        # Confusion matrix
        plt.subplot(1, 2, 2)
        cm = tf.math.confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['ham', 'spam'], 
                    yticklabels=['ham', 'spam'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.show()
        
        # Save model and tokenizer
        print("\nSaving model artifacts...")
        model.save('secure_spam_model.keras')
        
        with open('tokenizer.pkl', 'wb') as handle:
            pickle.dump(dataset.tokenizer, handle)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        if 'client' in locals():
            client._emergency_cleanup()

if __name__ == "__main__":
    main()
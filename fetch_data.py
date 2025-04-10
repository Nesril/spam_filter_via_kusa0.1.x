# fetch_data.py
import tensorflow as tf
from torch.utils.data import Dataset
import numpy as np
from kusa import DatasetClient, DatasetSDKException


class SecureSpamDataset(Dataset):
    def __init__(self, client: DatasetClient, batch_size: int):
        self.client = client
        self.batch_size = batch_size
        
        # Initialize dataset
        init_data = client.initialize()
        self.total_rows = init_data["totalRows"]
        self.max_batches = (self.total_rows // batch_size)
        
        # Initialize tokenizer with first batch
        first_batch = self._fetch_processed_batch(1)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(first_batch["text_samples"])

    def _fetch_processed_batch(self, batch_number):
        """Use SDK's secure processing"""
        def process_func(df):
            print("df ",df)
            return {
                "text_samples": df.iloc[:, 1].tolist(),  # Messages column
                "labels": [1 if x == "spam" else 0 for x in df.iloc[:, 0]]  # Labels column
            }
        result = self.client.fetch_and_process_batch(
            batch_size=self.batch_size,
            batch_number=batch_number,
            process_func=process_func
        )
        print()
                
        return result

    def __len__(self):
        return self.max_batches * self.batch_size

    def __getitem__(self, idx):
        batch_num = (idx // self.batch_size) + 1
        batch_data = self._fetch_processed_batch(batch_num)
        
        sample_idx = idx % self.batch_size
        text = batch_data["text_samples"][sample_idx]
        
        # Tokenize securely
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequence, 
            maxlen=100, 
            padding='post'
        )
        
        return padded[0], batch_data["labels"][sample_idx]

    def get_tf_dataset(self):
        """Create TensorFlow dataset without exposing raw data"""
        def generator():
            for batch_num in range(1, self.max_batches + 1):
                batch = self._fetch_processed_batch(batch_num)
                for text, label in zip(batch["text_samples"], batch["labels"]):
                    seq = self.tokenizer.texts_to_sequences([text])
                    padded = tf.keras.preprocessing.sequence.pad_sequences(
                        seq, maxlen=100, padding='post'
                    )
                    yield padded[0], label
        
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(100,), dtype=tf.int32),  # Text sequence
                tf.TensorSpec(shape=(), dtype=tf.int32)       # Label
            )
        ).batch(self.batch_size)
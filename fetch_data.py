import numpy as np
import tensorflow as tf
from torch.utils.data import Dataset
from kusa import DatasetClient, DatasetSDKException

class RemoteDataset(Dataset):
    def __init__(self, client: DatasetClient, batch_size: int):
        self.client = client
        self.batch_size = batch_size
        self.init_data = self.client.initialize()
        self.total_rows = self.init_data["totalRows"]
        self.current_batch = 1
        self.max_batches = (self.total_rows // batch_size)  # Floor division for complete batches

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.inputs, self.labels = self.fetch_batch(self.current_batch)
        self.tokenizer.fit_on_texts(self.inputs)  # Fit tokenizer on the entire input data

    def fetch_batch(self, batch_number):
        try:
            batch_data = self.client.fetch_batch(self.batch_size, batch_number)
            if batch_data.empty:
                raise RuntimeError(f"Batch {batch_number} is empty.")
            labels = batch_data.iloc[:, 0].values
            inputs = batch_data.iloc[:, 1].values  # Fetch the message column directly
            # print(labels,inputs)
            return inputs, labels
        except DatasetSDKException as e:
            raise RuntimeError(f"Failed to fetch batch {batch_number}: {e}")

    def __len__(self):
        return self.max_batches * self.batch_size

    def __getitem__(self, idx):
        batch_number = (idx // self.batch_size) + 1
        if batch_number != self.current_batch:
            self.inputs, self.labels = self.fetch_batch(batch_number)
            self.current_batch = batch_number

        sample_idx = idx % self.batch_size
        if sample_idx >= len(self.inputs):
            raise IndexError("Index out of range in the current batch.")

        input_text = self.inputs[sample_idx]
        input_vector = self.tokenizer.texts_to_sequences([input_text])  # Convert to sequence
        input_vector = tf.keras.preprocessing.sequence.pad_sequences(input_vector, padding='post')  # Pad sequences
        
        # Check the shape of the padded vector
        if input_vector.shape[0] == 0:
            raise ValueError("Padded input vector is empty.")
        
        return input_vector[0], self.labels[sample_idx]  # Return the padded input vector and the corresponding label

    def as_tf_dataset(self):
        all_inputs = []
        all_labels = []
        for batch_number in range(1, self.max_batches + 1):
            inputs, labels = self.fetch_batch(batch_number)
            all_inputs.append(inputs)
            all_labels.append(labels)
        inputs = np.concatenate(all_inputs)
        labels = np.concatenate(all_labels)
        inputs = self.tokenizer.texts_to_sequences(inputs)  # Convert all inputs to sequences
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, padding='post')  # Pad all inputs
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.batch(self.batch_size)
        return dataset

    def get_all_data(self):
        all_inputs = []
        all_labels = []
        for batch_number in range(1, self.max_batches + 1):
            inputs, labels = self.fetch_batch(batch_number)
            all_inputs.append(inputs)
            all_labels.append(labels)
        return np.concatenate(all_inputs), np.concatenate(all_labels)

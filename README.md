# Spam Filtering using kusa SDK

Welcome to the **Spam Filtering Application** built using the **kusa SDK**. This SDK was developed by **Dukana**, a data-driven eCommerce platform, to facilitate data acquisition for customers who purchase datasets. This tool provides a seamless interface for integrating Dukana's datasets into your **Python** applications, particularly for **Machine Learning (ML)** or **Deep Learning (DL)** projects.

## What is kusa SDK?

The **kusa SDK** is designed to enable developers to easily access and manage datasets that are purchased from the Dukana platform. This SDK currently supports dataset authorization, initialization, and batch fetching. It ensures that your application has smooth and efficient access to the dataset without the need to manually handle data retrieval processes.

The current version (v0.1.x) of the kusa SDK offers basic functionalities such as:
- **Authorization** to access the dataset.
- **Batch-based** dataset fetching.

> **Note:** The SDK is still in its early stages and does not yet support data encryption or enhanced security measures for datasets. However, improvements in data security are expected in future versions.

## Installation

Before using the kusa SDK, ensure that it is installed via pip:

```bash
pip install requirements.txt              

```

## Initialization

```bash
from kusa import DatasetClient, DatasetSDKException

client = DatasetClient(public_id=PUBLIC_ID, secret_key=SECRET_KEY)
try:
    init_data = client.initialize()
    print(f"Total Rows: {init_data['totalRows']}")
    print("First 10 Rows:")
    print(init_data['first10Rows'])
except DatasetSDKException as e:
    print(f"Initialization error: {e}")

```

## Fetch Data in Batches
Retrieve data in batches for processing:

```bash
batch_data = client.fetch_batch(batch_size=100, batch_number=1)
for record in batch_data:
    print(record)
```
### Expected Future Features From Kusa
  * Data Encryption to enhance security.
  * Advanced Authentication for dataset protection.

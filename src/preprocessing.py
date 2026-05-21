import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

class AmharicNewsDataset(torch.utils.data.Dataset):
    """Custom Dataset for Amharic news classification using Hugging Face tokenizer."""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize on the fly
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Squeeze out the batch dimension added by return_tensors="pt"
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item

def clean_data(df):
    """Preprocess data by selecting required columns, dropping NaNs, and combining text."""
    # Select only required columns
    df = df[["headline", "article", "category"]].copy()

    # Drop rows with missing values in any of the important columns
    df = df.dropna(subset=["headline", "article", "category"])

    # Ensure text columns are strings (important for concatenation)
    df["headline"] = df["headline"].astype(str)
    df["article"] = df["article"].astype(str)

    # Combine headline + article
    df["text"] = df["headline"] + " " + df["article"]

    # Reset index after dropping rows
    df = df.reset_index(drop=True)

    return df

def encode_labels(df, column="category"):
    """Encode string labels into integers."""
    df = df.dropna(subset=[column]).copy()
    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df[column])
    return df, label_encoder

def tokenize_data(texts, model_name="castorini/afriberta_large"):
    """Tokenize text using the specified Hugging Face model tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(
        list(texts),
        padding=True,
        truncation=True,
        max_length=512
    )

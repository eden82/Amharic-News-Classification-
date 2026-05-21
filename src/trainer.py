import os
import torch
import pickle
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.preprocessing import AmharicNewsDataset

def compute_metrics(eval_pred):
    """Compute standard multi-class classification metrics."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def train_model(df, label_encoder, model_name="castorini/afriberta_small", output_dir="./results", epochs=3, batch_size=8, lr=2e-5):
    """Split data, tokenize, initialize AfriBERTa sequence classifier, and train."""
    # Detect hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Hardware Detection] Using device: {device.upper()}")
    
    # Split dataset into train, validation, and test (80/10/10)
    # Stratify by labels to ensure equal distribution of categories
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])
    
    print(f"Dataset split sizes:")
    print(f"  Train:      {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test:       {len(test_df)} samples")
    
    # Load tokenizer
    print(f"\nLoading tokenizer for '{model_name}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create PyTorch datasets
    print("Wrapping subsets in PyTorch datasets...")
    train_dataset = AmharicNewsDataset(train_df["text"], train_df["label"], tokenizer)
    val_dataset = AmharicNewsDataset(val_df["text"], val_df["label"], tokenizer)
    test_dataset = AmharicNewsDataset(test_df["text"], test_df["label"], tokenizer)
    
    # Load model
    num_labels = len(label_encoder.classes_)
    print(f"\nLoading sequence classification model '{model_name}' with {num_labels} classes...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    
    # Define training arguments
    print("Setting up training configurations...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10 if len(train_df) > 100 else 2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",  # Avoid external integrations like wandb/tensorboard crashing on local run
        learning_rate=lr
    )
    
    # Initialize trainer
    print("Initializing Hugging Face Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training loop...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating model on unseen test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"\n--- Final Test Results ---")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Save the model and tokenizer
    model_save_path = os.path.join(output_dir, "best_model")
    print(f"\nSaving best model and tokenizer to: {model_save_path}")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # Save label encoder using pickle (essential for matching class index back to text name)
    encoder_path = os.path.join(model_save_path, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print("Label encoder successfully saved!")
    
    return trainer, test_results

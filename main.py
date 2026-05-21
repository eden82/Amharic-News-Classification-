import os
import torch
import pickle
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data_loader import load_data
from src.preprocessing import clean_data, encode_labels
from src.trainer import train_model

def main():
    file_path = "Amharic_News_Dataset.csv"
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        print("Please ensure the dataset is in the current directory.")
        return

    print("==================================================")
    print("      AMHARIC NEWS CLASSIFIER - AFRIBERTA         ")
    print("==================================================")

    print("\n1. Loading Data...")
    df = load_data(file_path)
    if df is None:
        print("Failed to load dataset. Exiting.")
        return
    print(f"   Initial dataset shape: {df.shape}")

    print("\n2. Data Analysis & Stats...")
    print(f"   Missing values count:\n{df.isnull().sum()}")
    print(f"   Category distribution:\n{df['category'].value_counts(dropna=False)}")

    print("\n3. Preprocessing Data...")
    df_clean = clean_data(df)
    print(f"   Cleaned dataset shape: {df_clean.shape}")
    
    print("\n4. Encoding Labels...")
    df_clean, label_encoder = encode_labels(df_clean)
    print(f"   Detected {len(label_encoder.classes_)} unique categories:")
    for idx, cat in enumerate(label_encoder.classes_):
        print(f"     [{idx}] -> {cat}")

    # Detect hardware resources and adjust defaults
    use_gpu = torch.cuda.is_available()
    
    print("\n5. Resource Configuration...")
    if use_gpu:
        print("   [+] GPU (CUDA) detected! Model training will run on full GPU acceleration.")
        model_name = "castorini/afriberta_base"
        epochs = 3
        batch_size = 16
        lr = 2e-5
        # For GPU, we can train on a larger fraction or full dataset
        sample_size = None
    else:
        print("   [!] WARNING: No GPU detected. Running on CPU.")
        print("   Training transformers on CPU is extremely slow. We will downsample")
        print("   the dataset to a small stratified subset (e.g. 500 samples total)")
        print("   and use a smaller AfriBERTa model version ('castorini/afriberta_small')")
        print("   so you can verify the pipeline runs successfully in a few minutes.")
        print("   If you have a GPU, please configure PyTorch to use CUDA.")
        
        # CPU config
        model_name = "castorini/afriberta_small"
        epochs = 1
        batch_size = 4
        lr = 2e-5
        sample_size = 500  # Will use 500 for training, 100 validation, 100 test

    if sample_size is not None:
        print(f"\n   [Downsampling] Extracting a stratified subset of {sample_size} rows...")
        # Stratified sampling to preserve category distribution
        samples_per_class = sample_size // len(label_encoder.classes_)
        df_clean = df_clean.groupby("category", group_keys=False).apply(
            lambda x: x.sample(min(len(x), samples_per_class), random_state=42)
        )
        # Shuffle
        df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"   Subsample size: {len(df_clean)} rows")

    print(f"\n6. Initializing Training with AfriBERTa...")
    print(f"   Model name:    {model_name}")
    print(f"   Epochs:        {epochs}")
    print(f"   Batch size:    {batch_size}")
    print(f"   Learning rate: {lr}")
    
    trainer, test_results = train_model(
        df=df_clean,
        label_encoder=label_encoder,
        model_name=model_name,
        output_dir="./results",
        epochs=epochs,
        batch_size=batch_size,
        lr=lr
    )

    print("\n7. Pipeline Completed Successfully!")
    
    # Simple Inference Test
    model_path = "./results/best_model"
    if os.path.exists(model_path):
        print("\n==================================================")
        print("        RUNNING SAMPLE INFERENCE TEST             ")
        print("==================================================")
        print(f"Loading saved model from {model_path} for quick inference...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Test Sample - Sport news text
        sample_text = "የኢትዮጵያ ቦክስ ፌዴሬሽን በየዓመቱ የሚያዘጋጀው የክለቦች ቻምፒዮና በአዲስ አበባ ከተማ በመካሄድ ላይ ይገኛል።"
        print(f"\nInput Amharic Text:\n  '{sample_text}'")
        
        # Tokenize & predict
        inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            pred_idx = torch.argmax(logits, dim=-1).item()
            
        predicted_category = label_encoder.inverse_transform([pred_idx])[0]
        print(f"Predicted News Category: {predicted_category}")
        print("==================================================")

if __name__ == "__main__":
    main()

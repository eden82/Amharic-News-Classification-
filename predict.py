import os
import sys
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def predict_category(text, model_path="./results/best_model"):
    """Load model, tokenizer, and label encoder to predict category of Amharic text."""
    if not os.path.exists(model_path):
        print(f"Error: Model not found at '{model_path}'. Please run training first!")
        return None
        
    # Load LabelEncoder
    encoder_path = os.path.join(model_path, "label_encoder.pkl")
    if not os.path.exists(encoder_path):
        print(f"Error: label_encoder.pkl not found in '{model_path}'")
        return None
        
    with open(encoder_path, "rb") as f:
        label_encoder = pickle.load(f)
        
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    # Preprocess and predict
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_idx = torch.argmax(logits, dim=-1).item()
        
    category = label_encoder.inverse_transform([pred_idx])[0]
    return category

if __name__ == "__main__":
    if len(sys.argv) > 1:
        text_to_predict = " ".join(sys.argv[1:])
    else:
        # Default sample news text
        text_to_predict = "የኢትዮጵያ ቦክስ ፌዴሬሽን በየዓመቱ የሚያዘጋጀው የክለቦች ቻምፒዮና በአዲስ አበባ ከተማ በመካሄድ ላይ ይገኛል።"
        print("No input text provided. Running prediction on a default sports news headline:")
        
    print(f"\nText: '{text_to_predict}'")
    category = predict_category(text_to_predict)
    if category:
        print(f"Predicted News Category: {category}\n")

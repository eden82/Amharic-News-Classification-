# 📰 Amharic News Classification using AfriBERTa

Welcome to the **Amharic News Classification using AfriBERTa** project repository!  
This project focuses on building an intelligent Natural Language Processing (NLP) system capable of automatically classifying Amharic news articles using the powerful **AfriBERTa transformer model**.

The repository contains datasets, preprocessing scripts, model training pipelines, evaluation results, and research implementations developed collaboratively by our project team members.

---

# 👥 Group Members

| Group Member | Student UGR ID |
| :--- | :---: |
| **Shalom Mesfin** | `UGR/25453/14` |
| **Benjamin Endale** | `UGR/25484/14` |
| **Salem Mesfin** | `UGR/25407/14` |
| **Yaikob Wasihun** | `UGR/25556/14` |
| **Bereket Daniel** | `UGR/25430/14` |

---

# 📌 Project Overview

This project applies **Deep Learning** and **Transformer-based NLP techniques** to classify Amharic news articles into 6 different categories automatically:
1. **ሀገር አቀፍ ዜና** (National News)
2. **ስፖርት** (Sports)
3. **ፖለቲካ** (Politics)
4. **ዓለም አቀፍ ዜና** (International News)
5. **ቢዝነስ** (Business)
6. **መዝናኛ** (Entertainment)

Using **AfriBERTa**, a multilingual transformer language model pre-trained specifically on 11 African languages (including Amharic), the system achieves state-of-the-art accuracy for sequence classification tasks on low-resource Amharic text.

---

# 📂 Project Structure

```
/home/yaikob-wasihun/Desktop/NLP/
├── Amharic_News_Dataset.csv   # Dataset file containing 51,483 news articles
├── requirements.txt           # Package dependencies
├── main.py                    # Main pipeline executor (training + evaluation + sample prediction)
├── predict.py                 # CLI inference utility for predictions on new Amharic texts
└── src/
    ├── __init__.py            # Module entrypoint exposing modules
    ├── data_loader.py         # CSV loader with inspection functions
    ├── preprocessing.py       # Cleaning text, encoding labels, and PyTorch dataset wrappers
    ├── trainer.py             # Model fine-tuning workflow using Hugging Face Trainer
    └── visualization.py       # Matplotlib category distribution plots
```

---

# ⚙️ Hardware & Resource Handling

Training large language models (LLMs) like AfriBERTa is highly resource-intensive. The training pipeline in `main.py` is engineered to detect your system resources dynamically:

- **GPU (CUDA Available)**: Automatically trains on the full dataset of **51,483 rows** using `castorini/afriberta_base` (or `castorini/afriberta_large`), employing larger batches and full GPU acceleration.
- **CPU (CUDA Not Available)**: Automatically extracts a **stratified representative subset** (e.g., 500 total samples) to preserve class distributions, and runs training using the lighter `castorini/afriberta_small` model. This enables the training pipeline to run successfully on any computer in just a few minutes for validation and testing purposes without hanging or running out of memory.

---

# 🚀 How to Run

### 1. Install Dependencies
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Training Pipeline
Run the main pipeline which loads data, performs cleaning/preprocessing, splits data into Train/Val/Test subsets, fine-tunes the AfriBERTa sequence classifier, evaluates on the test set, saves checkpoints, and runs a quick prediction test:
```bash
python3 main.py
```

### 3. Run Inference on New News Articles
Once the model is trained and saved under `results/best_model`, you can easily classify any new Amharic headlines or articles using the CLI tool:
```bash
python3 predict.py "የአፍሪካ ዋንጫ ማጣሪያ ውድድር በቅርቡ እንደሚጀመር የኢትዮጵያ እግር ኳስ ፌዴሬሽን አስታውቋል።"
```

---

# 🛠 Technologies Used

- **Python**
- **PyTorch**
- **Hugging Face Transformers** (AfriBERTa)
- **Scikit-learn**
- **Pandas & NumPy**
- **Matplotlib**
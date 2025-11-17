# 📄 Research Paper Recommendation & Subject Area Prediction

An **AI-powered application** for recommending research papers and predicting their subject areas using **MLP, Logistic Regression, and 1D-CNN models**. This project leverages **ArXiv abstracts dataset** and **Sentence Transformers** for semantic similarity.

---

## 🔗 Project Links
- Dataset: [ArXiv Paper Abstracts on Kaggle](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts/data)  
- Live Demo (if hosted): *Add your link here*  

---

## 🧠 Project Overview

This project provides:

1. **Paper Recommendations**: Based on semantic similarity using **Sentence Transformers** embeddings.
2. **Subject Area Predictions**: Multi-label classification using:
   - **Model 1**: Shallow Multi-Layer Perceptron (MLP) with TF-IDF vectors
   - **Model 2**: Logistic Regression (baseline)
   - **Model 3**: 1D Convolutional Neural Network (CNN) with word embeddings

**Why this project?**  
- Helps researchers quickly find relevant papers.
- Automatically predicts ArXiv subject areas from abstracts.
- Supports large-scale multi-label classification.

---

## 🗂 Dataset

The project uses the **ArXiv Paper Abstracts dataset**:

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts/data)
- **Columns**: `titles`, `abstracts`, `terms`
- **Preprocessing**:
  - Removed duplicates
  - Filtered rare terms
  - Explored distribution of abstract lengths and top subject areas
  - Created TF-IDF and integer-sequence representations for models

---

## 🛠 Features

### 1. Research Paper Recommendation
- Uses **Sentence Transformers** (`all-MiniLM-L6-v2`) for semantic embeddings.
- Computes **cosine similarity** between paper titles.
- Returns **top-5 recommended papers**.

### 2. Subject Area Prediction
- Multi-label classification for ArXiv subject categories.
- Uses three different models:
  - **MLP**: TF-IDF input, two hidden layers, dropout for regularization.
  - **Logistic Regression**: Baseline, one-vs-rest classifier.
  - **1D-CNN**: Embedding + Conv1D + GlobalMaxPooling for n-gram pattern detection.
- Threshold-based prediction with confidence scores.

---

## ⚙ Installation

```bash
# Clone repository
git clone <your-repo-url>
cd <your-repo-folder>

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
Key Libraries Used:

tensorflow

torch

sentence-transformers

scikit-learn

pandas, numpy, matplotlib, seaborn

streamlit (for app interface)

🚀 How to Run
Streamlit App
bash
Copy code
streamlit run app.py
Enter a paper title for recommendations

Enter an abstract to predict subject areas

Get top recommendations and predicted labels with confidence

Notebook
Open Research_Area_Prediction.ipynb

Run the cells sequentially:

Data loading & preprocessing

EDA & visualization

Model training (MLP, Logistic Regression, 1D-CNN)

Benchmarking & evaluation

Save & load models for deployment

📊 Model Evaluation
Model	Input Type	Accuracy / Metric	Notes
MLP	TF-IDF vectors	Binary Accuracy	Shallow non-linear model
Logistic Regression	TF-IDF vectors	F1-score (macro)	Baseline
1D-CNN	Integer sequences	Binary Accuracy	Captures n-gram patterns

The 1D-CNN model generally performs best on semantic patterns and multi-label classification.

📝 Example Predictions
Abstract:
"Graph neural networks are used for learning node representations."

Predicted Categories (Top-5):

cs.LG: 92.5%

cs.AI: 87.4%

stat.ML: 74.1%

cs.NE: 63.2%

cs.CV: 59.8%

📂 Saved Models & Files
models/model.keras → MLP model

models/model_cnn.keras → 1D-CNN model

models/log_reg_pipeline.pkl → Logistic Regression

models/text_vectorizer_config.pkl → TextVectorization config

models/text_vectorizer_weights.pkl → TF-IDF / vectorizer weights

models/label_vocab.pkl → Label vocabulary

models/embeddings.pkl → Sentence Transformer embeddings

models/sentences.pkl → Paper titles

models/rec_model.pkl → Sentence Transformer recommendation model

⚡ Limitations
Class imbalance in the dataset affects rare categories

Only predicts categories with > 1 occurrence in dataset

TF-IDF and basic embeddings may not capture deep semantic meaning

🌟 Future Improvements
Fine-tune BERT/SciBERT for better semantic understanding

Implement data augmentation for rare classes

Hyperparameter tuning for all models

Deploy a web interface with faster response using GPU acceleration

👥 Authors
**SYED WASIA ALI SHAH & ASHRAF MAHDI**

📄 License
This project is released under the MIT License

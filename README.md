📄 Research Area & Subject Area Prediction for ArXiv Papers






📝 Project Overview

This project provides a multi-label classification system for predicting research areas and subject areas of scientific papers from ArXiv. It also includes a recommendation engine to suggest similar papers based on titles using Sentence Transformers embeddings.

The project is implemented using:

Deep Learning Models: MLP (TF-IDF), 1D-CNN (embedding + convolution)

Classical ML: Logistic Regression (baseline)

Embedding-based Recommendations: Sentence Transformers

🔗 Dataset

The dataset contains ArXiv paper abstracts and metadata:

Source: Kaggle - ArXiv Paper Abstracts

Columns: titles, abstracts, terms (subject areas)

Preprocessing steps:

Remove duplicate titles

Filter rare terms (appear only once)

Text vectorization using TF-IDF (MLP & Logistic Regression) and integer sequences (1D-CNN)

Multi-label encoding using StringLookup (TensorFlow) and MultiLabelBinarizer (Scikit-learn)

🛠 Features

Subject Area Prediction

Multi-label classification for ArXiv abstracts

Models:

Shallow MLP: TF-IDF vectorized input

Logistic Regression: Baseline, TF-IDF input

1D-CNN: Embedding + Conv1D over sequences

Returns top categories with probabilities

Paper Recommendation

Generates top 5 similar papers based on title embeddings

Uses sentence-transformers (all-MiniLM-L6-v2)

Streamlit Web App

User-friendly interface

Input paper title for recommendations

Input abstract for subject area predictions

Displays predictions with confidence bars

📊 Exploratory Data Analysis

Abstract length distribution

Top 20 most frequent subject areas

Word cloud visualization for paper abstracts

Data cleaning & filtering for rare categories

⚙️ Installation

Clone the repository:

git clone https://github.com/<your-username>/arxiv-subject-prediction.git
cd arxiv-subject-prediction


Create a Python virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows


Install dependencies:

pip install -r requirements.txt


Required packages:

tensorflow==2.15.0

torch==2.0.1

sentence-transformers==2.2.2

streamlit

scikit-learn, pandas, numpy, matplotlib, seaborn, wordcloud

🚀 Usage
1️⃣ Jupyter Notebook

Run Arxiv_Subject_Prediction.ipynb for:

Data preprocessing

EDA (plots, word clouds)

Model training (MLP, Logistic Regression, 1D-CNN)

Evaluation and benchmarking

2️⃣ Streamlit App

Run the app:

streamlit run app.py


App Features:

Input paper title → get top 5 recommended papers

Input paper abstract → get predicted subject areas with confidence

Displays progress bars and probability scores

🏆 Model Architecture
1️⃣ Shallow MLP

Input: TF-IDF vector of abstracts

Layers: 512 → 256 → output (sigmoid for multi-label)

Loss: binary_crossentropy

Optimizer: adam

2️⃣ Logistic Regression

Baseline linear model

TF-IDF input

One-vs-Rest classification

3️⃣ 1D-CNN

Input: Integer-encoded sequences

Embedding layer → Conv1D → GlobalMaxPooling → Dense → Sigmoid

Captures local n-gram patterns

📈 Results
Model	Accuracy / F1	Notes
Shallow MLP	XX%	Good for frequent categories
Logistic Regression	XX%	Baseline, simple & fast
1D-CNN	XX%	Best performance, captures sequence info

⚠ Class imbalance affects rare categories. 1D-CNN performs best on frequent & medium-frequency classes.

🔮 Future Improvements

Fine-tune BERT / SciBERT for state-of-the-art performance

Address class imbalance with oversampling or augmentation

Hyperparameter tuning for CNN / MLP

Deploy web app with Docker / Streamlit Cloud

📂 File Structure
arxiv-subject-prediction/
├─ models/                   # Saved models and vectorizers
├─ Arxiv_Subject_Prediction.ipynb  # Notebook with preprocessing and training
├─ app.py                    # Streamlit app
├─ requirements.txt
├─ README.md
└─ dataset/
   └─ arxiv_data_210930-054931.csv

🧑‍💻 Authors

Ashraf Mahdi

Syed Wasia Ali Shah

🔗 References

Kaggle - ArXiv Paper Abstracts

TensorFlow Docs

Sentence Transformers

Streamlit

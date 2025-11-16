# pip install tensorflow==2.15.0
# pip install torch==2.0.1
# pip install sentence_transformers==2.2.2
# pip install streamlit

import streamlit as st
import torch
from sentence_transformers import util
import pickle
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
import numpy as np
from tensorflow import keras
import os

# Set page config.\.venv\Scripts\activate

st.set_page_config(
    page_title="Research Paper Recommender",
    page_icon="📄",
    layout="wide"
)

# Define paths (adjust these based on your directory structure)
MODEL_DIR = "models"  # Change this to your models directory path

# File paths
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, "embeddings.pkl")
SENTENCES_PATH = os.path.join(MODEL_DIR, "sentences.pkl")
REC_MODEL_PATH = os.path.join(MODEL_DIR, "rec_model.pkl")
PREDICTION_MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")  # Updated to .keras
TEXT_VECTORIZER_CONFIG_PATH = os.path.join(MODEL_DIR, "text_vectorizer_config.pkl")
TEXT_VECTORIZER_WEIGHTS_PATH = os.path.join(MODEL_DIR, "text_vectorizer_weights.pkl")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.pkl")
LABEL_VOCAB_PATH = os.path.join(MODEL_DIR, "label_vocab.pkl")
TRAIN_TEXTS_PATH = os.path.join(MODEL_DIR, "train_texts_for_idf.pkl")

# Load models with error handling===================================
@st.cache_resource
def load_recommendation_models():
    """Load recommendation models with error handling"""
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings = pickle.load(f)
        with open(SENTENCES_PATH, 'rb') as f:
            sentences = pickle.load(f)
        with open(REC_MODEL_PATH, 'rb') as f:
            rec_model = pickle.load(f)
        
        st.success("✓ Recommendation models loaded successfully!")
        return embeddings, sentences, rec_model
    except FileNotFoundError as e:
        st.error(f"❌ Error loading recommendation models: {e}")
        st.info(f"Make sure these files exist in '{MODEL_DIR}' directory:")
        st.write("- embeddings.pkl")
        st.write("- sentences.pkl")
        st.write("- rec_model.pkl")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None, None, None

@st.cache_resource
def load_prediction_models():
    """Load prediction models with error handling"""
    try:
        # Load the prediction model
        loaded_model = keras.models.load_model(PREDICTION_MODEL_PATH, compile=False)
        
        # Load text vectorizer configuration
        with open(TEXT_VECTORIZER_CONFIG_PATH, "rb") as f:
            saved_text_vectorizer_config = pickle.load(f)
        
        # Load vocabularies
        with open(VOCAB_PATH, "rb") as f:
            loaded_vocab = pickle.load(f)
        
        with open(LABEL_VOCAB_PATH, "rb") as f:
            label_vocab = pickle.load(f)
        
        # Create text vectorizer
        loaded_text_vectorizer = TextVectorization(
            max_tokens=saved_text_vectorizer_config['max_tokens'],
            ngrams=saved_text_vectorizer_config.get('ngrams', None),
            output_mode=saved_text_vectorizer_config['output_mode'],
            output_sequence_length=saved_text_vectorizer_config.get('output_sequence_length', None),
        )
        
        # Load weights if available, otherwise re-adapt
        try:
            with open(TEXT_VECTORIZER_WEIGHTS_PATH, "rb") as f:
                weights = pickle.load(f)
            
            if len(weights) == 0:
                # Re-adapt if no weights
                with open(TRAIN_TEXTS_PATH, "rb") as f:
                    train_texts = pickle.load(f)
                loaded_text_vectorizer.adapt(train_texts)
        except FileNotFoundError:
            st.warning("Weights file not found. Re-adapting vectorizer...")
            with open(TRAIN_TEXTS_PATH, "rb") as f:
                train_texts = pickle.load(f)
            loaded_text_vectorizer.adapt(train_texts)
        
        # Initialize the vectorizer
        _ = loaded_text_vectorizer(tf.constant(["test"], dtype=tf.string))
        
        st.success("✓ Prediction models loaded successfully!")
        return loaded_model, loaded_text_vectorizer, loaded_vocab, label_vocab
    
    except FileNotFoundError as e:
        st.error(f"❌ Error loading prediction models: {e}")
        st.info(f"Make sure these files exist in '{MODEL_DIR}' directory:")
        st.write("- model.keras")
        st.write("- text_vectorizer_config.pkl")
        st.write("- vocab.pkl")
        st.write("- label_vocab.pkl")
        st.write("- train_texts_for_idf.pkl")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None

# Load all models
embeddings, sentences, rec_model = load_recommendation_models()
loaded_model, loaded_text_vectorizer, loaded_vocab, label_vocab = load_prediction_models()

# Custom functions====================================
def recommendation(input_paper):
    """Generate paper recommendations based on input paper title"""
    if embeddings is None or rec_model is None or sentences is None:
        return ["Models not loaded. Please check file paths."]
    
    try:
        # Calculate cosine similarity
        cosine_scores = util.cos_sim(embeddings, rec_model.encode(input_paper))
        
        # Get top-k similar papers
        top_similar_papers = torch.topk(cosine_scores, dim=0, k=5, sorted=True)
        
        # Retrieve titles
        papers_list = []
        for i in top_similar_papers.indices:
            papers_list.append(sentences[i.item()])
        
        return papers_list
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return []

def invert_multi_hot(encoded_labels, vocab):
    """Reverse a single multi-hot encoded label to category names"""
    hot_indices = np.argwhere(encoded_labels == 1.0)[..., 0]
    return np.take(vocab, hot_indices)

def predict_category(abstract, model, vectorizer, label_vocab):
    """Predict subject areas for given abstract"""
    if model is None or vectorizer is None or label_vocab is None:
        return []
    
    try:
        # Preprocess the abstract
        input_data = tf.constant([abstract], dtype=tf.string)
        preprocessed_abstract = vectorizer(input_data)
        
        # Make predictions
        predictions = model.predict(preprocessed_abstract, verbose=0)
        
        # Convert to binary and get labels with probabilities
        pred_probs = predictions[0]
        binary_predictions = (pred_probs > 0.5).astype(int)
        
        # Get predicted indices
        pred_indices = np.where(binary_predictions == 1)[0]
        
        # Return labels with probabilities
        results = [(str(label_vocab[idx]), float(pred_probs[idx])) for idx in pred_indices]
        
        return results
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return []

# Create app=========================================
st.title('📄 Research Papers Recommendation and Subject Area Prediction')
st.markdown("### *LLM and Deep Learning Based Application*")

# Sidebar for information
with st.sidebar:
    st.header("About")
    st.write("This app provides:")
    st.write("- 📚 Paper recommendations based on title similarity")
    st.write("- 🏷️ Subject area predictions from abstracts")
    
    st.header("Model Status")
    if embeddings is not None and rec_model is not None:
        st.success("✓ Recommendation model loaded")
    else:
        st.error("✗ Recommendation model not loaded")
    
    if loaded_model is not None:
        st.success("✓ Prediction model loaded")
    else:
        st.error("✗ Prediction model not loaded")

# Main interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Input")
    input_paper = st.text_input("Enter Paper Title:", placeholder="e.g., Graph Neural Networks for Molecular Property Prediction")
    new_abstract = st.text_area(
        "Paste Paper Abstract:", 
        placeholder="Enter the full abstract of your paper here...",
        height=200
    )

with col2:
    st.subheader("ℹ️ Instructions")
    st.write("1. Enter a paper title for recommendations")
    st.write("2. Paste the paper abstract for subject area prediction")
    st.write("3. Click 'Analyze' to get results")

# Analyze button
if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not input_paper and not new_abstract:
        st.warning("⚠️ Please provide at least a paper title or abstract.")
    else:
        # Create tabs for results
        tab1, tab2 = st.tabs(["📚 Recommendations", "🏷️ Subject Areas"])
        
        with tab1:
            if input_paper:
                with st.spinner("Finding similar papers..."):
                    recommend_papers = recommendation(input_paper)
                    
                if recommend_papers:
                    st.subheader("Top 5 Recommended Papers")
                    for idx, paper in enumerate(recommend_papers, 1):
                        st.write(f"{idx}. {paper}")
                else:
                    st.info("No recommendations available.")
            else:
                st.info("Enter a paper title to get recommendations.")
        
        with tab2:
            if new_abstract:
                with st.spinner("Predicting subject areas..."):
                    predicted_categories = predict_category(
                        new_abstract, 
                        loaded_model, 
                        loaded_text_vectorizer, 
                        label_vocab
                    )
                
                if predicted_categories:
                    st.subheader("Predicted Subject Areas")
                    for label, prob in predicted_categories:
                        # Create a nice display with confidence bars
                        col_label, col_prob = st.columns([3, 1])
                        with col_label:
                            st.write(f"**{label}**")
                        with col_prob:
                            st.write(f"{prob*100:.1f}%")
                        st.progress(prob)
                else:
                    st.info("No predictions available or confidence too low.")
            else:
                st.info("Enter an abstract to predict subject areas.")

# Footer
st.markdown("---")
st.markdown("*Made by Ashraf Mahdi & Syed Wasia Ali Shah*")



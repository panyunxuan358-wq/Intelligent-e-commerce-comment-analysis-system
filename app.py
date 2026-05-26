import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import textwrap
import torch

# --- 1. Page Configuration ---
st.set_page_config(page_title="Intelligent E-commerce Review Analysis System", page_icon="🛍️")

# --- 2. Load Models ---
@st.cache_resource
def load_models():
    # Pipeline 1: Sentiment Analysis (Custom Fine-tuned Model)
    sentiment_pipe = pipeline("text-classification", model="panyunxuan/Model")
    
    # Pipeline 2: Summarization (Direct Loading to bypass system registration bugs)
    sum_model_name = "sshleifer/distilbart-cnn-12-6"
    sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)
    
    return sentiment_pipe, sum_tokenizer, sum_model

st.title("🛍️ Intelligent E-commerce Review Analysis System")
st.markdown("Input a product review, and AI will automatically determine the **Sentiment Orientation** and generate a **Core Summary**.")

with st.spinner("Initializing AI models, please wait..."):
    # Return three objects for manual inference
    sentiment_model, summary_tokenizer, summary_model = load_models()

# --- 3. Core Analysis Logic ---
def analyze_review(text):
    # Sentiment Analysis
    p1_res = sentiment_model(text[:512])[0]
    label = p1_res['label']
    score = p1_res['score']
    
    # Dynamic Summarization Logic (Manual Implementation)
    words = text.split()
    if len(words) > 30:
        # Calculate dynamic max_length based on input length
        d_max = max(25, min(int(len(words) * 0.3), 80))
        
        # 1. Manual Tokenization
        inputs = summary_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        
        # 2. Manual Summary Generation
        summary_ids = summary_model.generate(
            inputs["input_ids"], 
            max_length=d_max, 
            min_length=int(d_max/2), 
            early_stopping=True
        )
        
        # 3. Manual Decoding
        raw_s = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Sentence Integrity Protection
        last_p = max(raw_s.rfind('.'), raw_s.rfind('!'), raw_s.rfind('?'))
        final_s = raw_s[:last_p+1] if last_p != -1 else raw_s
    else:
        final_s = "Review is relatively short; core information is already in the original text."
        
    return label, score, final_s

# --- 4. Web Interaction Interface ---
input_text = st.text_area("Please paste an Amazon product review (English):", placeholder="Example: I love this phone case! It's very durable...", height=200)

if st.button("Analyze Now"):
    if input_text.strip():
        label, score, summary = analyze_review(input_text)
        st.divider()
        
        # Display Sentiment Results
        col1, col2 = st.columns(2)
        with col1:
            # Display color based on sentiment (LABEL_1 is Positive, LABEL_0 is Negative)
            color = "green" if label == "LABEL_1" else "red"
            sentiment_text = "Positive" if label == "LABEL_1" else "Negative"
            st.subheader("Sentiment Orientation")
            st.markdown(f":{color}[**{sentiment_text}**]")
            
        with col2:
            st.subheader("Confidence Score")
            st.progress(score)
            st.write(f"{score:.2%}")
            
        # Display Summary Results
        st.subheader("AI Summary")
        st.info(summary)
    else:
        st.warning("Please enter review content before clicking analyze.")

st.sidebar.markdown("---")
st.sidebar.info("Project Description: This system uses a fine-tuned DistilBERT for sentiment classification and a DistilBART for dynamic summarization.")
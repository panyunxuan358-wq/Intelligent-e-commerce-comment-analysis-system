import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import textwrap
import torch

# --- 1. 页面配置 ---
st.set_page_config(page_title="电商评论智能分析系统", page_icon="🛍️")

# --- 2. 加载模型 ---
@st.cache_resource
def load_models():
    # Pipeline 1: 情感分析 (保持不变，因为系统能识别它)
    sentiment_pipe = pipeline("text-classification", model="panyunxuan/Model")
    
    # Pipeline 2: 摘要分析 (改用底层加载，不使用 pipeline 注册系统)
    sum_model_name = "sshleifer/distilbart-cnn-12-6"
    sum_tokenizer = AutoTokenizer.from_pretrained(sum_model_name)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(sum_model_name)
    
    return sentiment_pipe, sum_tokenizer, sum_model

st.title("🛍️ Amazon 电商评论智能分析系统")
st.markdown("输入一段产品评论，AI 将自动判断**情感倾向**并生成**核心摘要**。")

with st.spinner("正在初始化 AI 模型，请稍候..."):
    # 注意这里返回了三个对象
    sentiment_model, summary_tokenizer, summary_model = load_models()

# --- 3. 核心分析逻辑 ---
def analyze_review(text):
    # 情感分析
    p1_res = sentiment_model(text[:512])[0]
    label = p1_res['label']
    score = p1_res['score']
    
    # 动态摘要逻辑 (底层实现)
    words = text.split()
    if len(words) > 30:
        d_max = max(25, min(int(len(words) * 0.3), 80))
        
        # 1. 手动分词
        inputs = summary_tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        
        # 2. 手动生成摘要
        summary_ids = summary_model.generate(
            inputs["input_ids"], 
            max_length=d_max, 
            min_length=int(d_max/2), 
            early_stopping=True
        )
        
        # 3. 手动解码
        raw_s = summary_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # 句子完整性保护
        last_p = max(raw_s.rfind('.'), raw_s.rfind('!'), raw_s.rfind('?'))
        final_s = raw_s[:last_p+1] if last_p != -1 else raw_s
    else:
        final_s = "评论较短，核心信息已在原文中。"
        
    return label, score, final_s

# --- 4. 网页交互界面 ---
input_text = st.text_area("请粘贴 Amazon 产品评论 (英文):", placeholder="Example: I love this phone case! It's very durable...", height=200)

if st.button("开始分析"):
    if input_text.strip():
        label, score, summary = analyze_review(input_text)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            color = "green" if label == "LABEL_1" else "red"
            sentiment_text = "正面 (Positive)" if label == "LABEL_1" else "负面 (Negative)"
            st.subheader("情感倾向")
            st.markdown(f":{color}[**{sentiment_text}**]")
        with col2:
            st.subheader("置信度")
            st.progress(score)
            st.write(f"{score:.2%}")
        st.subheader("内容精炼 (AI Summary)")
        st.info(summary)
    else:
        st.warning("请输入评论内容后再点击分析。")

st.sidebar.markdown("---")
st.sidebar.info("项目说明：本系统基于 DistilBERT 微调实现情感分类，并结合 DistilBART 实现动态摘要生成。")

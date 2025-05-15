import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

@st.cache_resource(show_spinner=False)
def load_model(model_path="./my_test_model"):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

tokenizer, model = load_model()

st.title("Customer Review Sentiment Classifier")
st.write("Masukkan ulasan pelanggan dan lihat prediksi sentimennya (positif/negatif).")

user_input = st.text_area("Tulis ulasan di sini:")

if st.button("Prediksi"):
    if user_input.strip() == "":
        st.warning("Masukkan teks ulasan terlebih dahulu!")
    else:
        inputs = tokenizer(user_input, return_tensors="pt")
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)
        label = torch.argmax(probs).item()
        labels_map = ["Negative", "Positive"]
        st.markdown(f"**Prediksi Sentimen:** {labels_map[label]}")
        st.markdown(f"**Confidence:** {probs[0,label].item():.2f}")

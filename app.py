<<<<<<< HEAD
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

st.title("📖 AI Summarizer App")
st.write("Enter any text below and get a concise summary using **T5-small** model.")

# Input text box
text_input = st.text_area("✍️ Paste your text here:", height=200)

# Summarize button
if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text to summarize.")
    else:
        # Prepare input for model
        input_ids = tokenizer.encode(
            "summarize: " + text_input,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Generate summary
        summary_ids = model.generate(
            input_ids,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode output
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("📝 Summary")
        st.success(summary)
=======
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
@st.cache_resource
def load_model():
    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.set_page_config(page_title="AI Text Summarizer", layout="wide")

st.title("📖 AI Summarizer App")
st.write("Enter any text below and get a concise summary using **T5-small** model.")

# Input text box
text_input = st.text_area("✍️ Paste your text here:", height=200)

# Summarize button
if st.button("Summarize"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text to summarize.")
    else:
        # Prepare input for model
        input_ids = tokenizer.encode(
            "summarize: " + text_input,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        # Generate summary
        summary_ids = model.generate(
            input_ids,
            max_length=150,
            min_length=40,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode output
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        st.subheader("📝 Summary")
        st.success(summary)
>>>>>>> bb28bdf0a52640545c74372b82e84434058abd6c

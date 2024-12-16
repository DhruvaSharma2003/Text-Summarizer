import streamlit as st
from transformers import pipeline

# Transformer-based Summarizer Initialization (Abstractive Summarization)
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load Extractive Summarizer (Using Hugging Face for extractive tasks)
extractive_summarizer = pipeline("feature-extraction", model="distilbert-base-uncased")

# Load Transformer for additional functionality (optional text generation)
text_generator = pipeline("text-generation", model="gpt2")

# Streamlit App Title
st.title("AI Text Summarizer")

# Input Text Area
input_text = st.text_area("Enter the text to summarize:")

# Select Summary Type
summary_type = st.selectbox("Select the summary type:", ["abstractive", "extractive", "generation"])

# Generate Summary Button
if st.button("Summarize"):
    if not input_text:
        st.error("Please enter some text to summarize.")
    else:
        try:
            if summary_type == "abstractive":
                # Perform Abstractive Summarization
                summary = abstractive_summarizer(input_text, max_length=130, min_length=30, do_sample=False)
                output = summary[0]['summary_text']
            elif summary_type == "extractive":
                # Perform Extractive Summarization (simplified feature extraction)
                embeddings = extractive_summarizer(input_text)
                output = f"Extracted Features Length: {len(embeddings[0])}"
            elif summary_type == "generation":
                # Perform Text Generation
                generated = text_generator(input_text, max_length=50, num_return_sequences=1)
                output = generated[0]['generated_text']
            else:
                st.error("Invalid summary type selected.")
                output = ""

            # Display Output
            st.subheader("Summary Result:")
            st.write(output)
        except Exception as e:
            st.error(f"Error: {str(e)}")

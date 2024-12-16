# AI Text Summarizer

This is a Flask-based AI Text Summarizer using Hugging Face Transformers. It provides:
1. **Abstractive Summarization** using `facebook/bart-large-cnn`.
2. **Extractive Summarization** using DistilBERT for feature extraction.
3. **Text Generation** using GPT-2.

## Features
- **Endpoint**: `/summarize`
- **Methods**: POST
- **Summary Types**:
  - `abstractive`: Generate human-like summaries.
  - `extractive`: Simplified feature extraction.
  - `generation`: Generate additional text using GPT-2.


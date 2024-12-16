# Importing Libraries
import os
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Initialize Flask App
app = Flask(__name__)

# Transformer-based Summarizer Initialization (Abstractive Summarization)
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load Extractive Summarizer (Using Hugging Face for extractive tasks)
extractive_summarizer = pipeline("feature-extraction", model="distilbert-base-uncased")

# Load Transformer for additional functionality (optional text generation)
text_generator = pipeline("text-generation", model="gpt2")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    if request.method == 'POST':
        data = request.json
        input_text = data.get("text")
        summary_type = data.get("type", "abstractive")

        if not input_text:
            return jsonify({"error": "No text provided."})

        try:
            if summary_type == "abstractive":
                # Perform abstractive summarization
                summary = abstractive_summarizer(input_text, max_length=130, min_length=30, do_sample=False)
                output = summary[0]['summary_text']
            elif summary_type == "extractive":
                # Perform extractive summarization (simplified feature extraction)
                embeddings = extractive_summarizer(input_text)
                output = f"Extracted Features Length: {len(embeddings[0])}"
            elif summary_type == "generation":
                # Additional feature: text generation
                generated = text_generator(input_text, max_length=50, num_return_sequences=1)
                output = generated[0]['generated_text']
            else:
                return jsonify({"error": "Invalid summary type."})

            return jsonify({"summary": output})
        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)

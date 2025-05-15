from flask import Flask, request, jsonify
import pdfplumber

app = Flask(__name__)

@app.route('/')
def home():
    return "Flask API is up!"

@app.route('/extract', methods=['POST'])
def extract():
    pdf = request.files.get('pdf')
    if not pdf:
        return jsonify({"error": "No file"}), 400
    text = ""
    with pdfplumber.open(pdf) as pdf_doc:
        for page in pdf_doc.pages:
            text += page.extract_text() or ""
    # For now, just return the first 500 chars
    return jsonify({"text": text[:500]})

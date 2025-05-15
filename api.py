import os
import re
import json
import pdfplumber
import openai
from flask import Flask, request, jsonify

# Initialize Flask app and OpenAI key
app = Flask(__name__)
openai.api_key = os.getenv('OPENAI_API_KEY')  # Set this in Render or environment

# Define keywords to extract
KEYWORDS = [
    "Total Assets",
    "Total Liabilities",
    "Intangible Assets",
    "Profit before Tax",
    "Cash and balances at central banks"
]

# Function to extract raw text and table rows from PDF
def extract_page_content(pdf_file):
    raw_text = []
    table_rows = []

    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Extract raw text
            text = page.extract_text() or ""
            raw_text.append(text)

            # Extract tables
            for table in page.extract_tables():
                for row in table:
                    table_rows.append(" | ".join(cell or "" for cell in row))

    return "\n".join(raw_text), table_rows

# Find keyword contexts in blocks of text
def find_contexts(text, keyword, window_chars=200):
    contexts = []
    for match in re.finditer(re.escape(keyword), text, re.IGNORECASE):
        start = max(0, match.start() - window_chars)
        end = match.end() + window_chars
        contexts.append(text[start:end])
    return contexts

# Find table rows containing the keyword
def find_table_rows(table_rows, keyword):
    return [row for row in table_rows if keyword.lower() in row.lower()]

# Prepare snippets for AI by combining contexts and relevant table rows
def prepare_snippets(raw_text, table_rows, max_snippets=20):
    snippets = []
    for kw in KEYWORDS:
        snippets.extend(find_contexts(raw_text, kw))
        snippets.extend(find_table_rows(table_rows, kw))
        if len(snippets) >= max_snippets:
            break
    return snippets[:max_snippets]

# Health check endpoint
@app.route('/')
def index():
    return "API is up!"

# Main extraction endpoint
@app.route('/extract', methods=['POST'])
def extract():
    pdf_file = request.files.get('pdf')
    if not pdf_file:
        return jsonify({"error": "No PDF file provided."}), 400

    # 1. Extract raw text and tables
    raw_text, table_rows = extract_page_content(pdf_file)

    # 2. Prepare snippets
    snippets = prepare_snippets(raw_text, table_rows)

    # 3. Query OpenAI for each keyword
    results = {}
    for kw in KEYWORDS:
        prompt = (
            f"Extract the value, unit, and year for '{kw}' from the text below. "
            "Respond only with JSON: {\"metric\":...,\"value\":...,\"unit\":...,\"year\":...}\n\n" +
            "\n---\n".join(snippets)
        )
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial data extractor. Provide only valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        text = response.choices[0].message.content.strip()
        try:
            results[kw] = json.loads(text)
        except json.JSONDecodeError:
            results[kw] = {"error": "Could not parse JSON", "raw": text}

    return jsonify(results)

# Allow direct execution
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

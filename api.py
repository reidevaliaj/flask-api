import os
import re
import json
import sqlite3
import tempfile
import fitz  # PyMuPDF for fast text scanning
import pdfplumber
import logging

from openai import OpenAI
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, flash
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ——————————————————————————————————————————————————————————————————————
#  A) Setup & config
# ——————————————————————————————————————————————————————————————————————
load_dotenv()  # load OPENAI_API_KEY & FLASK_SECRET from .env locally
app = Flask(__name__, static_folder='uploads')
app.secret_key = os.getenv('FLASK_SECRET', os.urandom(24))

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

UPLOAD_FOLDER = 'uploads'
DB_PATH       = 'extracted.db'
KEYWORDS      = [
    "Total Assets", "Total Liabilities", "Intangible Assets",
    "Profit before Tax", "Cash and balances at central banks"
]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logger.info("Initialized app with upload folder '%s' and DB '%s'", UPLOAD_FOLDER, DB_PATH)

# ——————————————————————————————————————————————————————————————————————
#  B) Database helpers (SQLite)
# ——————————————————————————————————————————————————————————————————————
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    db = get_db()
    db.execute(
        '''CREATE TABLE IF NOT EXISTS extracted_reports (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              filename TEXT NOT NULL,
              result_json TEXT NOT NULL,
              created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )'''
    )
    db.commit()
    logger.info("Database initialized and table ensured.")

init_db()

# ——————————————————————————————————————————————————————————————————————
#  C) Two-phase PDF extraction logic
# ——————————————————————————————————————————————————————————————————————
def find_relevant_pages(pdf_path, keywords):
    """
    Phase 1: fast, text-only scan using PyMuPD to flag pages containing keywords
    """
    logger.info("Phase 1: Scanning PDF '%s' for keywords %s", pdf_path, keywords)
    doc = fitz.open(pdf_path)
    keyset = [k.lower() for k in keywords]
    hits = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if any(k in text.lower() for k in keyset):
            hits.append(i)
    doc.close()
    logger.info("Phase 1 complete: found %d relevant pages: %s", len(hits), hits)
    return hits


def extract_page_content(pdf_path, hit_pages):
    """
    Phase 2: heavy parsing (text + tables) only on flagged pages via pdfplumber
    """
    logger.info("Phase 2: Parsing content on flagged pages of '%s'", pdf_path)
    raw_text = []
    table_rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx in hit_pages:
            if idx < len(pdf.pages):
                logger.info("Parsing page %d", idx)
                page = pdf.pages[idx]
                text = page.extract_text() or ""
                raw_text.append(text)
                for table in page.extract_tables():
                    for row in table:
                        table_rows.append(" | ".join(cell or "" for cell in row))
    logger.info("Phase 2 complete: extracted %d text blocks and %d table rows", len(raw_text), len(table_rows))
    return "\n".join(raw_text), table_rows


def find_contexts(text, keyword, window_chars=200):
    """Extracts text snippets around each occurrence of keyword"""
    contexts = []
    for m in re.finditer(re.escape(keyword), text, re.IGNORECASE):
        start = max(0, m.start() - window_chars)
        end = m.end() + window_chars
        contexts.append(text[start:end])
    return contexts


def find_table_rows(table_rows, keyword):
    """Filters table rows that contain the keyword"""
    return [row for row in table_rows if keyword.lower() in row.lower()]


def prepare_snippets(raw_text, table_rows, max_snippets=20):
    """Combines context snippets and table rows for AI input"""
    snippets = []
    for kw in KEYWORDS:
        snippets.extend(find_contexts(raw_text, kw))
        snippets.extend(find_table_rows(table_rows, kw))
        if len(snippets) >= max_snippets:
            break
    logger.info("Prepared %d snippets for AI extraction", len(snippets))
    return snippets[:max_snippets]


def call_ai(kw, snippets):
    """Sends snippets to OpenAI and returns parsed JSON result"""
    logger.info("Calling AI for keyword '%s' with %d snippets", kw, len(snippets))
    prompt = (
        f"Extract the value, unit & year for '{kw}' from the snippets below. "
        "Reply ONLY with JSON: {\"metric\":...,\"value\":...,\"unit\":...,\"year\":...}\n\n"
        + "\n---\n".join(snippets)
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial data extractor."},
            {"role": "user", "content": prompt}
        ]
    )
    text = resp.choices[0].message.content.strip()
    try:
        result = json.loads(text)
        logger.info("AI result for '%s': %s", kw, result)
        return result
    except json.JSONDecodeError:
        logger.error("Failed to parse JSON for '%s': %s", kw, text)
        return {"error": "Invalid JSON", "raw": text}

# ——————————————————————————————————————————————————————————————————————
#  D) Routes (upload, results, download)
# ——————————————————————————————————————————————————————————————————————
@app.route('/', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f:
            flash("Please select a PDF.")
            return redirect(url_for('upload'))

        filename = f.filename
        logger.info("Received upload for file '%s'", filename)

        # Save upload to a temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_path = tmp.name
        tmp.close()
        f.save(pdf_path)
        logger.info("Saved file to '%s'", pdf_path)

        # Two-phase extraction
        hit_pages = find_relevant_pages(pdf_path, KEYWORDS)
        if not hit_pages:
            os.remove(pdf_path)
            flash("No relevant pages found.")
            logger.warning("No pages matched keywords for '%s'", filename)
            return redirect(url_for('upload'))

        raw_text, table_rows = extract_page_content(pdf_path, hit_pages)
        snippets = prepare_snippets(raw_text, table_rows)
        results = {kw: call_ai(kw, snippets) for kw in KEYWORDS}

        # Store in DB
        db = get_db()
        db.execute("INSERT INTO extracted_reports (filename, result_json) VALUES (?, ?)",
                   (filename, json.dumps(results)))
        db.commit()
        rec_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.info("Stored results for '%s' as record %d", filename, rec_id)

        os.remove(pdf_path)
        return redirect(url_for('show_result', report_id=rec_id))

    # GET: show upload form + history
    db = get_db()
    db.row_factory = sqlite3.Row
    past = db.execute("SELECT * FROM extracted_reports ORDER BY created_at DESC").fetchall()
    return render_template('upload.html', past=past)

@app.route('/results/<int:report_id>')
def show_result(report_id):
    db = get_db()
    db.row_factory = sqlite3.Row
    rec = db.execute("SELECT * FROM extracted_reports WHERE id=?", (report_id,)).fetchone()
    data = json.loads(rec['result_json'])
    logger.info("Displaying results for record %d", report_id)
    return render_template('results.html', filename=rec['filename'], data=data)

@app.route('/uploads/<path:filename>')
def download_file(filename):
    logger.info("Download requested for '%s'", filename)
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    logger.info("Starting Flask app")
    app.run(host='0.0.0.0', port=5000, debug=True)

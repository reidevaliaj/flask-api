import os
import re
import json
import sqlite3
import tempfile
import fitz  # PyMuPDF for text extraction
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
load_dotenv()  # load OPENAI_API_KEY & FLASK_SECRET from .env
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
logger.info("App initialized: upload folder '%s', DB '%s'", UPLOAD_FOLDER, DB_PATH)

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
    logger.info("Database ready.")

init_db()

# ——————————————————————————————————————————————————————————————————————
#  C) Two-phase PDF extraction using PyMuPDF only
# ——————————————————————————————————————————————————————————————————————
def find_relevant_pages(pdf_path, keywords):
    """
    Phase 1: quick text scan to flag pages with keywords
    """
    logger.info("Phase 1 scan: %s", pdf_path)
    doc = fitz.open(pdf_path)
    key_low = [k.lower() for k in keywords]
    hits = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if any(kw in text.lower() for kw in key_low):
            hits.append(i)
    doc.close()
    logger.info("Flagged pages: %s", hits)
    return hits


def extract_page_content(pdf_path, hit_pages):
    """
    Phase 2: extract text from flagged pages
    """
    logger.info("Phase 2 parse: pages %s", hit_pages)
    doc = fitz.open(pdf_path)
    raw_text_parts = []
    for idx in hit_pages:
        if idx < doc.page_count:
            page_text = doc.load_page(idx).get_text() or ""
            raw_text_parts.append(page_text)
            logger.info("Extracted text from page %d, %d chars", idx, len(page_text))
    doc.close()
    raw_text = "\n".join(raw_text_parts)
    return raw_text

# ——————————————————————————————————————————————————————————————————————
#  D) Snippet extraction & AI call
# ——————————————————————————————————————————————————————————————————————
def find_contexts(text, keyword, window_chars=200):
    snippets = []
    for m in re.finditer(re.escape(keyword), text, re.IGNORECASE):
        s = max(0, m.start() - window_chars)
        e = m.end() + window_chars
        snippets.append(text[s:e])
    return snippets


def prepare_snippets(raw_text, keywords, max_snippets=20):
    all_snips = []
    for kw in keywords:
        ctxs = find_contexts(raw_text, kw)
        all_snips.extend(ctxs)
        if len(all_snips) >= max_snippets:
            break
    logger.info("Prepared %d context snippets", len(all_snips))
    return all_snips[:max_snippets]


def call_ai(kw, snippets):
    logger.info("AI call for '%s' with %d snippets", kw, len(snippets))
    prompt = (
        f"Extract the value, unit & year for '{kw}' from these snippets. "
        "Reply only with JSON: {\"metric\":...,\"value\":...,\"unit\":...,\"year\":...}\n\n"
        + "\n---\n".join(snippets)
    )
    resp = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a financial data extractor."},
            {"role": "user", "content": prompt}
        ]
    )
    content = resp.choices[0].message.content.strip()
    try:
        result = json.loads(content)
        logger.info("AI result for '%s': %s", kw, result)
        return result
    except json.JSONDecodeError:
        logger.error("JSON parse error for '%s': %s", kw, content)
        return {"error": "Invalid JSON", "raw": content}

# ——————————————————————————————————————————————————————————————————————
#  E) Routes: upload, results, download
# ——————————————————————————————————————————————————————————————————————
@app.route('/', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('pdf')
        if not f:
            flash("Please select a PDF.")
            return redirect(url_for('upload'))
        fname = f.filename
        logger.info("Uploading %s", fname)

        # save to temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        path = tmp.name
        tmp.close()
        f.save(path)

        # extract
        pages = find_relevant_pages(path, KEYWORDS)
        if not pages:
            flash("No keywords found.")
            os.remove(path)
            return redirect(url_for('upload'))
        raw = extract_page_content(path, pages)
        snippets = prepare_snippets(raw, KEYWORDS)
        results = {kw: call_ai(kw, snippets) for kw in KEYWORDS}

        # store
        db = get_db()
        db.execute("INSERT INTO extracted_reports (filename, result_json) VALUES (?, ?)",
                   (fname, json.dumps(results)))
        db.commit()
        rec_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.info("Stored record %d for %s", rec_id, fname)

        os.remove(path)
        return redirect(url_for('show_result', report_id=rec_id))
    db = get_db()
    rows = db.execute("SELECT * FROM extracted_reports ORDER BY created_at DESC").fetchall()
    return render_template('upload.html', past=rows)

@app.route('/results/<int:report_id>')
def show_result(report_id):
    db = get_db()
    rec = db.execute("SELECT * FROM extracted_reports WHERE id=?", (report_id,)).fetchone()
    data = json.loads(rec['result_json'])
    return render_template('results.html', filename=rec['filename'], data=data)

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    logger.info("Starting Flask on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
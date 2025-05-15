import os
import re
import json
import sqlite3
import tempfile
import fitz  # PyMuPDF for fast text scanning
import pdfplumber
from openai import OpenAI
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, flash
from dotenv import load_dotenv

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

init_db()

# ——————————————————————————————————————————————————————————————————————
#  C) Two-phase PDF extraction logic with PyMuPDF and pdfplumber
# ——————————————————————————————————————————————————————————————————————

def find_relevant_pages(pdf_path, keywords):
    """
    Phase 1: fast, text-only scan using PyMuPDF to flag pages containing keywords
    """
    doc = fitz.open(pdf_path)
    keyset = [k.lower() for k in keywords]
    hits = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if any(k in text.lower() for k in keyset):
            hits.append(i)
    return hits


def extract_page_content(pdf_path, hit_pages):
    """
    Phase 2: heavy parsing (text + tables) only on flagged pages via pdfplumber
    """
    raw_text = []
    table_rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx in hit_pages:
            if idx < len(pdf.pages):
                page = pdf.pages[idx]
                text = page.extract_text() or ""
                raw_text.append(text)
                for table in page.extract_tables():
                    for row in table:
                        table_rows.append(" | ".join(cell or "" for cell in row))
    return "\n".join(raw_text), table_rows


def find_contexts(text, keyword, window_chars=200):
    out = []
    for m in re.finditer(re.escape(keyword), text, re.IGNORECASE):
        s = max(0, m.start() - window_chars)
        e = m.end() + window_chars
        out.append(text[s:e])
    return out


def find_table_rows(table_rows, keyword):
    return [r for r in table_rows if keyword.lower() in r.lower()]


def prepare_snippets(raw_text, table_rows, max_snippets=20):
    snippets = []
    for kw in KEYWORDS:
        snippets.extend(find_contexts(raw_text, kw))
        snippets.extend(find_table_rows(table_rows, kw))
        if len(snippets) >= max_snippets:
            break
    return snippets[:max_snippets]


def call_ai(kw, snippets):
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
        return json.loads(text)
    except json.JSONDecodeError:
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

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        pdf_path = tmp.name
        tmp.close()
        f.save(pdf_path)

        # Two-phase extraction
        hit_pages = find_relevant_pages(pdf_path, KEYWORDS)
        if not hit_pages:
            os.remove(pdf_path)
            flash("No relevant pages found.")
            return redirect(url_for('upload'))

        raw_text, table_rows = extract_page_content(pdf_path, hit_pages)
        snippets = prepare_snippets(raw_text, table_rows)
        results = {kw: call_ai(kw, snippets) for kw in KEYWORDS}

        # Store in DB
        db = get_db()
        db.execute(
          "INSERT INTO extracted_reports (filename, result_json) VALUES (?, ?)",
          (f.filename, json.dumps(results))
        )
        db.commit()
        rec_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        os.remove(pdf_path)
        return redirect(url_for('show_result', report_id=rec_id))

    # GET: show upload form + past reports
    db = get_db()
    past = db.execute("SELECT * FROM extracted_reports ORDER BY created_at DESC").fetchall()
    return render_template('upload.html', past=past)

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
    app.run(host='0.0.0.0', port=5000, debug=True)
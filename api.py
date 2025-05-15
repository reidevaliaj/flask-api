import os, re, json, sqlite3, pdfplumber, openai
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory, flash
from dotenv import load_dotenv

# ——————————————————————————————————————————————————————————————————————
#  A) Setup & config
# ——————————————————————————————————————————————————————————————————————
load_dotenv()  # load OPENAI_API_KEY from .env locally
app = Flask(__name__, static_folder='uploads')
app.secret_key = os.getenv('FLASK_SECRET', 'supersecret')

openai.api_key = os.getenv('OPENAI_API_KEY')

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
#  C) PDF extraction logic (your existing code)
# ——————————————————————————————————————————————————————————————————————
def extract_page_content(pdf_path):
    raw_text, table_rows = [], []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            raw_text.append(txt)
            for table in page.extract_tables():
                for row in table:
                    table_rows.append(" | ".join(cell or "" for cell in row))
    return "\n".join(raw_text), table_rows

def find_contexts(text, kw, window=200):
    out = []
    for m in re.finditer(re.escape(kw), text, re.IGNORECASE):
        s = max(0, m.start()-window)
        e = m.end()+window
        out.append(text[s:e])
    return out

def find_table_rows(rows, kw):
    return [r for r in rows if kw.lower() in r.lower()]

def prepare_snippets(raw, rows, maxn=20):
    snippets = []
    for kw in KEYWORDS:
        snippets += find_contexts(raw, kw)
        snippets += find_table_rows(rows, kw)
        if len(snippets)>=maxn:
            break
    return snippets[:maxn]

def call_ai(kw, snippets):
    prompt = (
      f"Extract the value, unit & year for “{kw}” from the snippets below. "
      "Reply ONLY with JSON like "
      '{"metric": "...", "value": "...", "unit": "...", "year": ...}\\n\\n'
      + "\n---\n".join(snippets)
    )
    resp = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {"role":"system","content":"You are a financial data extractor."},
        {"role":"user","content":prompt}
      ]
    )
    txt = resp.choices[0].message.content.strip()
    try:
        return json.loads(txt)
    except:
        return {"error":"parse_failed","raw":txt}

# ——————————————————————————————————————————————————————————————————————
#  D) Routes
# ——————————————————————————————————————————————————————————————————————
@app.route('/', methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files.get('pdf')
        if not f:
            flash("Please select a PDF.")
            return redirect(url_for('upload'))

        # 1) Save PDF
        save_path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(save_path)

        # 2) Extract
        raw, rows  = extract_page_content(save_path)
        snippets   = prepare_snippets(raw, rows)
        results    = {kw: call_ai(kw, snippets) for kw in KEYWORDS}

        # 3) Store in DB
        db = get_db()
        db.execute(
          "INSERT INTO extracted_reports (filename, result_json) VALUES (?, ?)",
          (f.filename, json.dumps(results))
        )
        db.commit()
        report_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

        return redirect(url_for('show_result', report_id=report_id))

    # GET → show upload form + list of past PDFs
    db = get_db()
    past = db.execute("SELECT * FROM extracted_reports ORDER BY created_at DESC").fetchall()
    return render_template('upload.html', past=past)

@app.route('/results/<int:report_id>')
def show_result(report_id):
    db = get_db()
    rec = db.execute("SELECT * FROM extracted_reports WHERE id=?",(report_id,)).fetchone()
    data = json.loads(rec['result_json'])
    return render_template('results.html', filename=rec['filename'], data=data)

@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

# ——————————————————————————————————————————————————————————————————————
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000)

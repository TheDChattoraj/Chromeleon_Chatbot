import os
import logging
from flask import Flask, request, jsonify, render_template
from src.rag.rag_runner import RAGRunner
from src.ingest.indexer import Indexer
from werkzeug.utils import secure_filename
from pathlib import Path
import io
from flask import send_file, request, current_app, abort
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from src.app.config import PERSIST_DIR, EMBEDDING_MODEL


app = Flask(__name__)

# Create a single global runner instance (keeps vectorstore/chain in memory)
RAG = RAGRunner()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

indexer = Indexer(uploaded_path= UPLOAD_DIR, persist_dir=PERSIST_DIR, embedding_model=EMBEDDING_MODEL, delete_after_index=True)

def allowed_file(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


def render_kb_page_to_pdf(kb_number: str, timeout: int = 20000) -> bytes:
    """
    Use Playwright (Chromium) to load the KB page and render it to PDF.
    Returns raw PDF bytes. Raises on error.
    """
    url = f"https://resource.digital.thermofisher.com/kb/article.aspx?n={kb_number}"
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])  # consider headless options
        page = browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout)
        except PlaywrightTimeoutError:
            # try again or continue — best-effort
            page.goto(url, wait_until="load", timeout=timeout)

        # Optionally click "Printable View" if present. This often yields a cleaner print layout.
        # Use a short try/except because selector may not exist or text may differ.
        try:
            # attempt to click a button/link with text "Printable View"
            # adjust if KB site markup is different
            page.locator('text="Printable View"').click(timeout=2000)
            # wait for navigation / print view to render
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            # ignore if printable view not available
            pass

        # Render PDF: you can tweak format, margin, scale, etc.
        pdf_bytes = page.pdf(format="A4", print_background=True, margin={"top":"12mm","bottom":"12mm","left":"10mm","right":"10mm"})
        browser.close()
    return pdf_bytes


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Expected JSON:
    {
      "question": "string",
      "chat_history": [["user msg","assistant reply"], ...]  # optional
    }
    """
    payload = request.get_json() or {}
    question = payload.get("question", "").strip()
    chat_history = payload.get("chat_history", [])
    debug = payload.get("debug", False)

    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        # call RAGRunner.answer()
        result = RAG.answer(question, chat_history=chat_history, debug=debug)
        # Attach the developer-local file path so front-end/tooling can convert to URL if needed.
        
        return jsonify(result)
    except Exception as e:
        logging.exception("Error answering question: %s", e)
        return jsonify({"error": "internal error", "detail": str(e)}), 500
    

@app.route("/upload", methods=["POST"])
def upload_files():
    """
    Accepts multipart/form-data files under key 'files' (one or many).
    Returns JSON with per-file indexing results.
    """
    if "files" not in request.files:
        return jsonify({"error": "No files part in the request. Send files under key 'files'."}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files selected"}), 400

    results = []
    for f in files:
        filename = secure_filename(f.filename)
        if not filename:
            results.append({"file": None, "status": "failed", "error": "Invalid filename"})
            continue
        if not allowed_file(filename):
            results.append({"file": filename, "status": "failed", "error": "File type not allowed"})
            continue

        dest = UPLOAD_DIR / filename
        try:
            # Save temporarily to disk so loader can read it
            f.save(dest)

            # Call the indexer for THIS file path (pass string path)
            res = indexer.index_file_to_vectorstore(str(dest))
            # ensure file paths are strings (defensive)
            if isinstance(res.get("file"), (Path,)):
                res["file"] = str(res["file"])
            results.append(res)
            logging.info("Successfully indexed the document to Vector Store: %s", filename)
        except Exception as e:
            logging.exception("Indexing/upload failed for %s: %s", filename, e)
            results.append({"file": filename, "status": "failed", "error": str(e)})

    return jsonify({"results": results}), 200


@app.route("/download_kb")
def download_kb():
    """
    Example: /download_kb?kb=000064909
    Renders KB page to PDF and returns as attachment.
    """
    kb = request.args.get("kb", "").strip()
    if not kb or not kb.isdigit():
        return abort(400, "kb query parameter required (digits only)")

    try:
        pdf_bytes = render_kb_page_to_pdf(kb)
    except Exception as e:
        current_app.logger.exception("Failed to render KB %s to PDF: %s", kb, e)
        return abort(500, "Failed to render KB to PDF")

    # Stream the PDF back as a download
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"KB_{kb}.pdf",
    )
    



if __name__ == "__main__":
    # default host=127.0.0.1 and port=5000; change for your environment
    app.run(host="0.0.0.0", port=5000, debug=True)
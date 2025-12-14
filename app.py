# app.py
import os
import logging
import time
import random
import smtplib
import ssl
from datetime import timedelta
from functools import wraps
from pathlib import Path
import io

from dotenv import load_dotenv
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    flash,
    session,
    send_file,
    current_app,
    abort,
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

# RAG and indexer (your project-specific imports)
from src.rag.rag_runner import RAGRunner
from src.ingest.indexer import Indexer
from src.app.config import PERSIST_DIR, EMBEDDING_MODEL

# Playwright for KB -> PDF
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# Forms (your WTForms)
from src.login.form import EmailForm, OTPForm

# Redis / sessions / limiter
import redis
from flask_session import Session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask import make_response

load_dotenv()

# ---------------------------
# App & project setup
# ---------------------------
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Create a single global runner instance (keeps vectorstore/chain in memory)
RAG = RAGRunner()

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# ---------------------------
# Redis setup
# ---------------------------
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
# Important: do NOT decode responses globally because Flask-Session stores binary (pickled) session data
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)

# Ping Redis at startup to help debugging
try:
    redis_client.ping()
    app.logger.info("Connected to Redis at %s", REDIS_URL)
except Exception as e:
    app.logger.exception("Failed to connect to Redis (%s) - continuing but app may fail at runtime", e)

# Enable server-side sessions in Redis (recommended for multi-worker setups)
app.config["SESSION_TYPE"] = "redis"
app.config["SESSION_REDIS"] = redis_client
app.config["SESSION_PERMANENT"] = False
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=2)
Session(app)

# ---------------------------
# Rate limiter
# ---------------------------
# Note: pass get_remote_address (function) not get_remote_address()
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["10 per minute"])

# ---------------------------
# Gmail / SMTP config (for SMTPLIB)
# ---------------------------
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS") or os.environ.get("MAIL_USERNAME")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD") or os.environ.get("MAIL_PASSWORD")

# ---------------------------
# OTP / app configuration (ensure ints)
# ---------------------------
def _int_env(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        app.logger.warning("Environment variable %s invalid integer '%s'; using default %s", name, v, default)
        return default

OTP_TTL_SECONDS = _int_env("OTP_TTL_SECONDS", 300)
OTP_LENGTH = _int_env("OTP_LENGTH", 6)
MAX_OTP_ATTEMPTS = _int_env("MAX_OTP_ATTEMPTS", 5)

app.logger.info("OTP config: TTL=%s, LENGTH=%s, MAX_ATTEMPTS=%s", OTP_TTL_SECONDS, OTP_LENGTH, MAX_OTP_ATTEMPTS)

# ---------------------------
# Indexer instance
# ---------------------------
indexer = Indexer(uploaded_path=UPLOAD_DIR, persist_dir=PERSIST_DIR, embedding_model=EMBEDDING_MODEL, delete_after_index=True)

# ---------------------------
# Helpers: OTP generation, redis keys, send email
# ---------------------------
def generate_otp(n=OTP_LENGTH) -> str:
    start = 10 ** (n - 1)
    return str(random.randrange(start, start * 10))

def send_otp_gmail(to_email: str, otp: str):
    """
    Send an OTP using Gmail via smtplib. Requires GMAIL_ADDRESS and GMAIL_APP_PASSWORD in env.
    """
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        raise RuntimeError("Gmail credentials are not configured (GMAIL_ADDRESS/GMAIL_APP_PASSWORD)")

    subject = "Your OTP Login Code"
    body = f"Your one-time login code is: {otp}\nThis code expires in {OTP_TTL_SECONDS // 60} minutes."
    message = f"Subject: {subject}\n\n{body}"

    context = ssl.create_default_context()
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls(context=context)
        server.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        server.sendmail(GMAIL_ADDRESS, to_email, message)

def redis_otp_key(email: str) -> str:
    return f"otp:{email}"

def redis_attempts_key(email: str) -> str:
    return f"otp_attempts:{email}"

def store_otp(email: str, otp_plain: str):
    """Store hashed OTP (string) in Redis with TTL and reset attempts to 0."""
    otp_hash = generate_password_hash(otp_plain)  # returns a str
    key_otp = redis_otp_key(email)
    key_attempts = redis_attempts_key(email)
    p = redis_client.pipeline()
    # store the string encoded as utf-8 bytes (redis_client with decode_responses=False uses bytes)
    p.set(key_otp, otp_hash.encode("utf-8"), ex=OTP_TTL_SECONDS)
    p.set(key_attempts, 0, ex=OTP_TTL_SECONDS)
    p.execute()

def get_otp_hash(email: str):
    """Return the OTP hash as a str (decode bytes) or None."""
    val = redis_client.get(redis_otp_key(email))
    if val is None:
        return None
    # val is bytes (since decode_responses=False)
    if isinstance(val, bytes):
        try:
            return val.decode("utf-8")
        except Exception:
            # fallback: attempt latin1 or return None
            try:
                return val.decode("latin1")
            except Exception:
                return None
    return str(val)

def increment_attempts(email: str) -> int:
    key = redis_attempts_key(email)
    p = redis_client.pipeline()
    p.incr(key)
    p.expire(key, OTP_TTL_SECONDS)
    result = p.execute()[0]  # this may be int or bytes depending on redis client
    try:
        return int(result)
    except Exception:
        # If bytes, decode then int
        if isinstance(result, bytes):
            return int(result.decode())
        return int(result)

def clear_otp_records(email: str):
    redis_client.delete(redis_otp_key(email))
    redis_client.delete(redis_attempts_key(email))

# ---------------------------
# Authentication helpers
# ---------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_email"):
            # store original path so user can be redirected after login
            session['next'] = request.path
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped

# ---------------------------
# Routes: auth
# ---------------------------
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute")
def login():
    form = EmailForm()
    if form.validate_on_submit():
        email = form.email.data.lower()

        # Generate OTP, store in Redis, send via Gmail
        otp = generate_otp()
        try:
            store_otp(email, otp)
            send_otp_gmail(email, otp)
        except Exception as e:
            app.logger.exception("Failed sending OTP or storing in Redis: %s", e)
            flash("Failed to send OTP. Check mail settings or try again later.", "danger")
            return redirect(url_for("login"))

        session["pending_email"] = email
        flash("A code has been sent to your email. Check your inbox.", "info")
        return redirect(url_for("verify_otp"))

    return render_template("login.html", form=form)

@app.route("/verify-otp", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def verify_otp():
    form = OTPForm()
    pending_email = session.get("pending_email")
    if not pending_email:
        flash("Please enter your work email first.", "warning")
        return redirect(url_for("login"))

    otp_hash = get_otp_hash(pending_email)
    if not otp_hash:
        # expired or not found
        session.pop("pending_email", None)
        flash("OTP expired or not found. Request a new code.", "warning")
        return redirect(url_for("login"))

    if form.validate_on_submit():
        entered = form.otp.data.strip()

        attempts = increment_attempts(pending_email)
        if attempts > MAX_OTP_ATTEMPTS:
            clear_otp_records(pending_email)
            session.pop("pending_email", None)
            flash("Too many attempts. Request a new code.", "danger")
            return redirect(url_for("login"))

        if check_password_hash(otp_hash, entered):
            # Successful login
            clear_otp_records(pending_email)
            session.pop("pending_email", None)
            session["user_email"] = pending_email
            session["authenticated_at"] = time.time()
            flash("Login successful.", "success")
            next_url = session.pop("next", None) or url_for("index")
            return redirect(next_url)
        else:
            flash("Invalid code, try again.", "danger")

    return render_template("verify_otp.html", form=form, email=pending_email)

@app.route("/resend-otp")
@limiter.limit("3 per hour")
def resend_otp():
    pending = session.get("pending_email")
    if not pending:
        flash("No pending email, please enter your email.", "warning")
        return redirect(url_for("login"))

    otp = generate_otp()
    try:
        store_otp(pending, otp)
        send_otp_gmail(pending, otp)
        flash("A new code was sent to your email.", "info")
    except Exception as e:
        app.logger.exception("Failed to resend OTP: %s", e)
        flash("Failed to resend code. Try again later.", "danger")

    return redirect(url_for("verify_otp"))


@app.route("/logout", methods=["GET", "POST"])
def logout():
    """
    Logs out the user. Accepts GET (user click) and POST (sendBeacon).
    Returns 302 redirect for GET, 204 No Content for POST.
    """
    session.pop("user_email", None)
    session.pop("authenticated_at", None)
    # Also clear pending email if any
    session.pop("pending_email", None)

    # For POST (sendBeacon), return a 204 to avoid redirect issues
    if request.method == "POST":
        return ("", 204)

    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# ---------------------------
# Main app routes
# ---------------------------
# Protect index so user must login to access the app UI
@app.route("/")
@login_required
def index():
    email = session.get("user_email")
    username = email.split("@")[0] if email else ""
    # capitalize first letter
    pretty_name = username.replace(".", " ").title()
    first_name = pretty_name.split(" ")[0]
    # render your main app UI (index.html)
    return render_template("index.html", email = first_name)

# Keep the old 'protected' welcome page if you still want it
@app.route("/welcome")
@login_required
def welcome():
    return render_template("protected.html", user_email=session.get("user_email"))

# ---------------------------
# File upload / RAG / KB download etc (kept from original)
# ---------------------------
def allowed_file(filename: str) -> bool:
    return filename.lower().endswith(".pdf")

def render_kb_page_to_pdf(kb_number: str, timeout: int = 20000) -> bytes:
    """
    Use Playwright (Chromium) to load the KB page and render it to PDF.
    Returns raw PDF bytes. Raises on error.
    """
    url = f"https://resource.digital.thermofisher.com/kb/article.aspx?n={kb_number}"
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        try:
            page.goto(url, wait_until="networkidle", timeout=timeout)
        except PlaywrightTimeoutError:
            page.goto(url, wait_until="load", timeout=timeout)

        try:
            page.locator('text="Printable View"').click(timeout=2000)
            page.wait_for_load_state("networkidle", timeout=5000)
        except Exception:
            pass

        pdf_bytes = page.pdf(format="A4", print_background=True, margin={"top":"12mm","bottom":"12mm","left":"10mm","right":"10mm"})
        browser.close()
    return pdf_bytes

@app.route("/download_kb")
def download_kb():
    kb = request.args.get("kb", "").strip()
    if not kb or not kb.isdigit():
        return abort(400, "kb query parameter required (digits only)")

    try:
        pdf_bytes = render_kb_page_to_pdf(kb)
    except Exception as e:
        current_app.logger.exception("Failed to render KB %s to PDF: %s", kb, e)
        return abort(500, "Failed to render KB to PDF")

    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"KB_{kb}.pdf",
    )

@app.route("/api/query", methods=["POST"])
def api_query():
    payload = request.get_json() or {}
    question = payload.get("question", "").strip()
    chat_history = payload.get("chat_history", [])
    debug = payload.get("debug", False)

    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        result = RAG.answer(question, chat_history=chat_history, debug=debug)
        return jsonify(result)
    except Exception as e:
        logging.exception("Error answering question: %s", e)
        return jsonify({"error": "internal error", "detail": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_files():
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
            f.save(dest)
            res = indexer.index_file_to_vectorstore(str(dest))
            if isinstance(res.get("file"), Path):
                res["file"] = str(res["file"])
            results.append(res)
            app.logger.info("Successfully indexed the document: %s", filename)
        except Exception as e:
            app.logger.exception("Indexing/upload failed for %s: %s", filename, e)
            results.append({"file": filename, "status": "failed", "error": str(e)})

    return jsonify({"results": results}), 200

# ---------------------------
# Misc / index route shadow guard: keep only one index route above
# ---------------------------

# ---------------------------
# Run server
# ---------------------------
if __name__ == "__main__":
    # Use host 0.0.0.0 for containerized environments. Change debug=False for production.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)

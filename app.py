# app.py
"""
Iris Project - Professional, production-style app.py
- Robust auth (login/register/OTP reset), sessions, admin
- Prediction via JSON API and HTML form (ui.html)
- History (download, clear), dashboard stats, chart APIs
- New pages: analytics, dataset explorer, model info, notifications, help
- Admin: users list
- Safe file uploads for profile photos

Assumptions:
- Templates in ./templates
- Static uploads under ./static/uploads
- Dataset file at ./data/iris.csv
- Model/scaler in ./model/iris_model.pkl and ./model/scaler.pkl
"""

import os
import uuid
import csv
import hashlib
import random
from io import StringIO, BytesIO
from datetime import datetime, timedelta
from email.message import EmailMessage

from flask import (
    Flask, request, jsonify, render_template, redirect, url_for,
    session, send_file, flash, abort
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

import pickle
import mysql.connector
import numpy as np
import pandas as pd

# bcrypt optional (used to validate $2b$ hashes if present in DB)
try:
    import bcrypt as _bcrypt  # alias to avoid name clash
    bcrypt = _bcrypt
except Exception:
    bcrypt = None

# ---------- CONFIG ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

MODEL_DIR = os.path.join(BASE_DIR, "model")

UPLOAD_FOLDER = os.path.join(STATIC_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB

# Flask app
app = Flask(__name__)
CORS(app)
app.secret_key = os.environ.get("FLASK_SECRET", "supersecret123")  # change in production
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE
app.permanent_session_lifetime = timedelta(days=7)

# SMTP - optional (set env vars to enable)
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", 587) or 587)
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
EMAIL_FROM = os.environ.get("EMAIL_FROM", "noreply@example.com")

OTP_EXPIRY_MINUTES = int(os.environ.get("OTP_EXPIRY_MINUTES", 10))

# DB defaults (override with env vars)
DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "user": os.environ.get("DB_USER", "root"),
    "password": os.environ.get("DB_PASS", "Sanika123"),
    "database": os.environ.get("DB_NAME", "iris_project"),
    "autocommit": False,
}


# ---------- MODEL LOADING ----------
model = None
scaler = None
model_accuracy = None  # stored from dataset evaluation (best-effort)
try:
    model_path = os.path.join(MODEL_DIR, "iris_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        app.logger.info("Model & scaler loaded successfully.")
    else:
        app.logger.info("Model/scaler not present — prediction route will work after you add them.")
except Exception:
    app.logger.exception("Failed to load model/scaler")
    model = None
    scaler = None


# ---------- DB helper ----------
def db_connection():
    """Return a new mysql.connector connection using DB_CONFIG"""
    return mysql.connector.connect(**DB_CONFIG)


# ---------- Utilities ----------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def gen_token():
    return uuid.uuid4().hex


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def require_login() -> bool:
    return bool(session.get("logged_in") and session.get("user_id"))


def require_admin() -> bool:
    return bool(session.get("is_admin"))


def get_user_by_id(user_id):
    try:
        conn = db_connection()
        cur = conn.cursor(dictionary=True, buffered=True)
        cur.execute("SELECT * FROM users WHERE id=%s LIMIT 1", (user_id,))
        user = cur.fetchone()
        return user
    except Exception:
        app.logger.exception("get_user_by_id error")
        return None
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


def get_user_by_username(username):
    try:
        conn = db_connection()
        cur = conn.cursor(dictionary=True, buffered=True)
        cur.execute("SELECT * FROM users WHERE username=%s LIMIT 1", (username,))
        user = cur.fetchone()
        return user
    except Exception:
        app.logger.exception("get_user_by_username error")
        return None
    finally:
        try:
            cur.close()
            conn.close()
        except Exception:
            pass


def send_email(to_email: str, subject: str, body: str) -> bool:
    """Send email via SMTP if configured; otherwise print to console for development."""
    if SMTP_HOST and SMTP_USER and SMTP_PASS:
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = EMAIL_FROM
            msg["To"] = to_email
            msg.set_content(body)
            with __import__("smtplib").SMTP(SMTP_HOST, SMTP_PORT) as s:
                s.starttls()
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
            app.logger.info("Email sent to %s", to_email)
            return True
        except Exception:
            app.logger.exception("Email send failed")
            return False
    else:
        # Dev fallback
        print(f"[EMAIL FALLBACK] To: {to_email}\nSubject: {subject}\n\n{body}")
        return True


# ---------- ROUTES: AUTH ----------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        if not username or not password:
            return render_template("login.html", error="Username and password required")

        try:
            conn = db_connection()
            cur = conn.cursor(dictionary=True, buffered=True)
            cur.execute("SELECT * FROM users WHERE username=%s LIMIT 1", (username,))
            user = cur.fetchone()
            if not user:
                return render_template("login.html", error="Invalid username or password")

            stored = (user.get("password") or "").strip()
            authenticated = False

            # 1) werkzeug hash (pbkdf2, scrypt, argon2)
            if stored.startswith("pbkdf2:") or stored.startswith("scrypt:") or stored.startswith("argon2:"):
                try:
                    authenticated = check_password_hash(stored, password)
                except Exception:
                    authenticated = False

            # 2) bcrypt ($2b$...) if library available
            if not authenticated and bcrypt and stored.startswith("$2"):
                try:
                    authenticated = bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
                except Exception:
                    authenticated = False

            # 3) legacy SHA256 hex
            if not authenticated:
                if len(stored) == 64 and all(c in "0123456789abcdef" for c in stored.lower()):
                    if sha256_hex(password) == stored.lower():
                        authenticated = True
                        # upgrade password to werkzeug hash (best-effort)
                        try:
                            new_hash = generate_password_hash(password)
                            cur.execute("UPDATE users SET password=%s WHERE id=%s", (new_hash, user["id"]))
                            conn.commit()
                        except Exception:
                            pass

            # 4) plaintext fallback (very unlikely)
            if not authenticated and stored == password:
                authenticated = True
                try:
                    new_hash = generate_password_hash(password)
                    cur.execute("UPDATE users SET password=%s WHERE id=%s", (new_hash, user["id"]))
                    conn.commit()
                except Exception:
                    pass

            if not authenticated:
                return render_template("login.html", error="Invalid username or password")

            # success: set session
            token = gen_token()
            try:
                cur.execute("UPDATE users SET session_token=%s, last_login=CURRENT_TIMESTAMP WHERE id=%s", (token, user["id"]))
                conn.commit()
            except Exception:
                pass

            session.permanent = True
            session["logged_in"] = True
            session["username"] = user["username"]
            session["user_id"] = user["id"]
            session["session_token"] = token
            session["is_admin"] = bool(user.get("is_admin"))

            # log login activity (best-effort)
            try:
                ip = request.remote_addr
                ua = request.headers.get("User-Agent", "")[:512]
                cur.execute(
                    "INSERT INTO login_activity (user_id, ip, user_agent, logged_at) VALUES (%s,%s,%s,CURRENT_TIMESTAMP)",
                    (user["id"], ip, ua)
                )
                conn.commit()
            except Exception:
                pass

            return redirect("/dashboard")
        except Exception:
            app.logger.exception("Login error")
            return render_template("login.html", error="Internal error")
        finally:
            try:
                cur.close(); conn.close()
            except Exception:
                pass

    # GET
    return render_template("login.html")


@app.route("/profile")
def profile():
    if not require_login():
        return redirect("/login")
    user = get_user_by_id(session["user_id"])
    return render_template("profile.html", user=user)




@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        full_name = request.form.get("full_name") or username
        email = request.form.get("email") or None

        if not username or not password:
            return render_template("register.html", error="Username and password are required")

        try:
            conn = db_connection()
            cur = conn.cursor(dictionary=True, buffered=True)
            # check duplicates
            cur.execute("SELECT id FROM users WHERE username=%s OR email=%s LIMIT 1", (username, email))
            if cur.fetchone():
                return render_template("register.html", error="Username or email already exists")

            hashed_password = generate_password_hash(password)
            cur = conn.cursor(buffered=True)
            cur.execute(
                "INSERT INTO users (username, email, password, full_name, is_admin, created_at) VALUES (%s,%s,%s,%s,0,CURRENT_TIMESTAMP)",
                (username, email, hashed_password, full_name)
            )
            conn.commit()
            flash("Account created. Please login.")
            return redirect("/login")
        except mysql.connector.IntegrityError:
            return render_template("register.html", error="Username or email already exists")
        except Exception:
            app.logger.exception("Register error")
            return render_template("register.html", error="Database error")
        finally:
            try:
                cur.close(); conn.close()
            except Exception:
                pass

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/logout-all")
def logout_all():
    if not require_login():
        return redirect("/login")
    try:
        conn = db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE users SET session_token=NULL WHERE id=%s", (session.get("user_id"),))
        conn.commit()
        cur.close(); conn.close()
    except Exception:
        app.logger.exception("logout-all failed")
    session.clear()
    return redirect("/login")


@app.route("/delete-account", methods=["POST"])
def delete_account():
    if not require_login():
        return redirect("/login")
    uid = session.get("user_id")
    try:
        conn = db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE id=%s", (uid,))
        conn.commit()
        cur.close(); conn.close()
    except Exception:
        app.logger.exception("delete account failed")
        flash("Failed to delete account")
        return redirect("/settings")
    session.clear()
    flash("Account deleted")
    return redirect("/register")


# ---------- PASSWORD RESET (OTP) ----------
@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        identifier = (request.form.get("identifier") or "").strip()
        if not identifier:
            return render_template("forgot_password.html", error="Provide username or email")
        try:
            conn = db_connection()
            cur = conn.cursor(dictionary=True, buffered=True)
            cur.execute("SELECT * FROM users WHERE username=%s OR email=%s LIMIT 1", (identifier, identifier))
            user = cur.fetchone()
            if not user:
                return render_template("forgot_password.html", error="No user found")
            otp = f"{random.randint(100000, 999999)}"
            token = gen_token()
            expires_at = datetime.utcnow() + timedelta(minutes=OTP_EXPIRY_MINUTES)
            cur.execute("INSERT INTO password_reset_tokens (user_id, otp, token, expires_at, used, created_at) VALUES (%s,%s,%s,%s,0,CURRENT_TIMESTAMP)",
                        (user["id"], otp, token, expires_at))
            conn.commit()
            body = f"Your Iris Project OTP is: {otp}. Expires in {OTP_EXPIRY_MINUTES} minutes."
            if user.get("email"):
                send_email(user["email"], "Password reset OTP", body)
                method = "email"
            else:
                send_email("console", "Password reset OTP", body)
                method = "console"
            return render_template("forgot_password_sent.html", method=method, user=user)
        except Exception:
            app.logger.exception("forgot-password")
            return render_template("forgot_password.html", error="Internal error")
        finally:
            try:
                cur.close(); conn.close()
            except Exception:
                pass
    return render_template("forgot_password.html")


@app.route("/verify-otp", methods=["GET", "POST"])
def verify_otp():
    if request.method == "POST":
        identifier = (request.form.get("identifier") or "").strip()
        otp = (request.form.get("otp") or "").strip()
        if not identifier or not otp:
            return render_template("verify_otp.html", error="Provide identifier and OTP")
        try:
            conn = db_connection()
            cur = conn.cursor(dictionary=True, buffered=True)
            cur.execute("""
                SELECT t.*, u.username FROM password_reset_tokens t
                JOIN users u ON u.id = t.user_id
                WHERE (u.username=%s OR u.email=%s) AND t.otp=%s AND t.used=0 AND t.expires_at > CURRENT_TIMESTAMP
                ORDER BY t.created_at DESC LIMIT 1
            """, (identifier, identifier, otp))
            row = cur.fetchone()
            if not row:
                return render_template("verify_otp.html", error="Invalid or expired OTP")
            cur.execute("UPDATE password_reset_tokens SET used=1 WHERE id=%s", (row["id"],))
            conn.commit()
            token = row["token"]
            return redirect(url_for("reset_password", token=token))
        except Exception:
            app.logger.exception("verify-otp")
            return render_template("verify_otp.html", error="Internal error")
        finally:
            try:
                cur.close(); conn.close()
            except Exception:
                pass
    return render_template("verify_otp.html")


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if request.method == "POST":
        new_password = request.form.get("new_password") or ""
        if not new_password:
            return render_template("reset_password.html", token=token, error="Provide a new password")
        try:
            conn = db_connection()
            cur = conn.cursor(dictionary=True, buffered=True)
            cur.execute("SELECT * FROM password_reset_tokens WHERE token=%s AND expires_at > CURRENT_TIMESTAMP LIMIT 1", (token,))
            row = cur.fetchone()
            if not row:
                return render_template("reset_password.html", token=token, error="Token invalid or expired")
            new_hash = generate_password_hash(new_password)
            cur.execute("UPDATE users SET password=%s WHERE id=%s", (new_hash, row["user_id"]))
            cur.execute("UPDATE password_reset_tokens SET used=1 WHERE id=%s", (row["id"],))
            conn.commit()
            return redirect("/login")
        except Exception:
            app.logger.exception("reset-password")
            return render_template("reset_password.html", token=token, error="Internal error")
        finally:
            try:
                cur.close(); conn.close()
            except Exception:
                pass
    return render_template("reset_password.html", token=token)


# ---------- PAGES ----------
@app.route("/")
def index_route():
    if session.get("user_id"):
        return redirect("/dashboard")
    return render_template("index.html")


@app.route("/home")
def home_route():
    if not require_login():
        return redirect("/login")
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    if not require_login():
        return redirect("/login")
    user = get_user_by_id(session["user_id"])
    if not user:
        session.clear()
        return redirect("/login")

    total_predictions = 0
    last_pred = None
    class_counts = {"Setosa": 0, "Versicolor": 0, "Virginica": 0}

    try:
        conn = db_connection()
        cur = conn.cursor(dictionary=True, buffered=True)
        cur.execute("SELECT COUNT(*) AS c FROM prediction_history WHERE user_id=%s", (user["id"],))
        row = cur.fetchone()
        total_predictions = int(row["c"]) if row and row["c"] is not None else 0

        cur.execute("""
            SELECT predicted_class, COUNT(*) AS cnt
            FROM prediction_history WHERE user_id=%s GROUP BY predicted_class
        """, (user["id"],))
        rows = cur.fetchall()
        for r in rows:
            cls = r["predicted_class"]
            cnt = int(r["cnt"])
            if cls in class_counts:
                class_counts[cls] = cnt

        cur.execute("""
            SELECT predicted_class, confidence, timestamp
            FROM prediction_history WHERE user_id=%s ORDER BY timestamp DESC LIMIT 1
        """, (user["id"],))
        last_pred = cur.fetchone()
    except Exception:
        app.logger.exception("dashboard stat")
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

    return render_template(
    "dashboard.html",
    username=session.get("username"),
    total_predictions=total_predictions,
    last_pred_class=last_pred["predicted_class"] if last_pred else "—",
    last_confidence=round(last_pred["confidence"], 2) if last_pred else "—",
    last_timestamp=last_pred["timestamp"].strftime("%Y-%m-%d %H:%M") if last_pred else "—"
)



@app.route("/iris")
@app.route("/ui")
def ui_page():
    if not require_login():
        return redirect("/login")
    return render_template("ui.html")


# ---------- Prediction ----------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not require_login():
        return redirect("/login")

    if request.method == "GET":
        return render_template("predict.html")  # Show the form

    # POST method: handle prediction
    if model is None or scaler is None:
        return render_template("predict.html", error="Model not loaded on server")

    try:
        sepal_length = float(request.form.get("sepal_length"))
        sepal_width = float(request.form.get("sepal_width"))
        petal_length = float(request.form.get("petal_length"))
        petal_width = float(request.form.get("petal_width"))

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        scaled = scaler.transform(features)
        pred = model.predict(scaled)[0]
        confidence = 0.0
        try:
            proba = model.predict_proba(scaled)[0]
            confidence = float(max(proba)) * 100.0
        except Exception:
            confidence = 0.0

        # Save to DB
        try:
            conn = db_connection()
            cur = conn.cursor()
            user_id = session.get("user_id")
            cur.execute(
                "INSERT INTO prediction_history (user_id,sepal_length,sepal_width,petal_length,petal_width,predicted_class,confidence) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (user_id, sepal_length, sepal_width, petal_length, petal_width, str(pred), confidence)
            )
            conn.commit()
        except Exception:
            app.logger.exception("saving prediction")
        finally:
            try:
                cur.close(); conn.close()
            except Exception:
                pass

        return render_template("predict.html", prediction=f"Predicted: {pred} (Confidence: {round(confidence,2)}%)")
    except Exception:
        app.logger.exception("prediction error")
        return render_template("predict.html", error="Prediction failed")




# ---------- History ----------
@app.route("/history")
def history():
    if not require_login():
        return redirect("/login")
    try:
        conn = db_connection()
        cur = conn.cursor(dictionary=True, buffered=True)
        cur.execute("SELECT id,sepal_length,sepal_width,petal_length,petal_width,predicted_class,confidence,timestamp FROM prediction_history WHERE user_id=%s ORDER BY timestamp DESC", (session.get("user_id"),))
        history_data = cur.fetchall()
    except Exception:
        app.logger.exception("history fetch")
        history_data = []
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass
    return render_template("history.html", history=history_data, username=session.get("username"))


@app.route("/clear-history", methods=["POST"])
def clear_history():
    if not require_login():
        return redirect("/login")
    try:
        conn = db_connection()
        cur = conn.cursor()
        if session.get("is_admin"):
            cur.execute("TRUNCATE TABLE prediction_history")
        else:
            cur.execute("DELETE FROM prediction_history WHERE user_id=%s", (session.get("user_id"),))
        conn.commit()
        flash("Prediction history cleared.")
    except Exception:
        app.logger.exception("clear history")
        flash("Failed to clear history.")
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass
    return redirect("/history")


@app.route("/download-history")
def download_history():
    if not require_login():
        return redirect("/login")
    try:
        conn = db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id,sepal_length,sepal_width,petal_length,petal_width,predicted_class,confidence,timestamp FROM prediction_history WHERE user_id=%s ORDER BY timestamp DESC", (session.get("user_id"),))
        rows = cur.fetchall()
        cur.close(); conn.close()
    except Exception:
        app.logger.exception("download history")
        rows = []

    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(["id","sepal_length","sepal_width","petal_length","petal_width","predicted_class","confidence","timestamp"])
    for r in rows:
        writer.writerow(r)

    csv_bytes = si.getvalue().encode("utf-8")
    bio = BytesIO(csv_bytes)
    bio.seek(0)

    # support older Flask versions
    try:
        return send_file(
            bio,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    except TypeError:
        return send_file(
            bio,
            mimetype="text/csv",
            as_attachment=True,
            attachment_filename=f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )


# ---------- Chart data ----------
@app.route("/chart-data")
def chart_data():
    if not require_login():
        return jsonify({})
    try:
        conn = db_connection()
        cur = conn.cursor()
        cur.execute("SELECT predicted_class, COUNT(*) FROM prediction_history WHERE user_id=%s GROUP BY predicted_class", (session.get("user_id"),))
        rows = cur.fetchall()
    except Exception:
        app.logger.exception("chart-data")
        rows = []
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

    data = {"Setosa": 0, "Versicolor": 0, "Virginica": 0}
    for item in rows:
        try:
            cls = item[0]
            cnt = int(item[1])
        except Exception:
            continue
        if cls in data:
            data[cls] = cnt
    return jsonify(data)


# ---------- SETTINGS & PROFILE ----------
@app.route("/settings", methods=["GET"])
def settings():
    if not require_login():
        return redirect("/login")
    user = get_user_by_id(session["user_id"])
    activity = []
    history_rows = []
    try:
        conn = db_connection()
        cur = conn.cursor(dictionary=True, buffered=True)
        cur.execute("SELECT * FROM login_activity WHERE user_id=%s ORDER BY logged_at DESC LIMIT 20", (user["id"],))
        activity = cur.fetchall()

        cur.execute("SELECT id AS id, predicted_class AS prediction, confidence, timestamp FROM prediction_history WHERE user_id=%s ORDER BY timestamp DESC LIMIT 50", (user["id"],))
        history_rows = cur.fetchall()
    except Exception:
        app.logger.exception("settings data")
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass
    return render_template("settings.html", user=user or {}, activity=activity, history=history_rows)


@app.route("/update-settings", methods=["POST"])
def update_settings():
    if not require_login():
        return redirect("/login")
    user_id = session.get("user_id")
    new_username = request.form.get("new_username")
    new_password = request.form.get("new_password")
    full_name = request.form.get("full_name")
    email = request.form.get("email")
    notifications = 1 if request.form.get("notifications") == "on" else 0
    theme = request.form.get("theme") or "light"

    photo_filename = None
    if "profile_photo" in request.files:
        f = request.files["profile_photo"]
        if f and f.filename and allowed_file(f.filename):
            original = secure_filename(f.filename)
            prefix = uuid.uuid4().hex[:8]
            photo_filename = f"{prefix}_{original}"
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], photo_filename)
            f.save(save_path)
            # try to delete old
            try:
                user = get_user_by_id(user_id)
                old = user.get("profile_photo") if user else None
                if old:
                    old_path = os.path.join(app.config["UPLOAD_FOLDER"], old)
                    if os.path.exists(old_path):
                        os.remove(old_path)
            except Exception:
                pass

    fields = []
    values = []
    if new_username:
        fields.append("username=%s"); values.append(new_username)
    if new_password:
        fields.append("password=%s"); values.append(generate_password_hash(new_password))
    if full_name is not None:
        fields.append("full_name=%s"); values.append(full_name)
    if email is not None:
        fields.append("email=%s"); values.append(email)
    fields.append("notifications=%s"); values.append(notifications)
    fields.append("theme=%s"); values.append(theme)
    if photo_filename:
        fields.append("profile_photo=%s"); values.append(photo_filename)

    if not fields:
        flash("No changes provided.")
        return redirect("/settings")

    values.append(user_id)
    query = f"UPDATE users SET {', '.join(fields)} WHERE id=%s"
    try:
        conn = db_connection()
        cur = conn.cursor(buffered=True)
        cur.execute(query, tuple(values))
        conn.commit()
        cur.close(); conn.close()
    except Exception:
        app.logger.exception("update-settings")
        flash("Update failed.")
        try:
            cur.close(); conn.close()
        except Exception:
            pass

    if new_username:
        session["username"] = new_username
    flash("Settings updated.")
    return redirect("/settings")


@app.route("/delete-photo", methods=["POST"])
def delete_photo():
    if not require_login():
        return redirect("/login")
    user_id = session.get("user_id")
    user = get_user_by_id(user_id)
    profile_photo = user.get("profile_photo") if user else None
    if profile_photo:
        try:
            path = os.path.join(app.config["UPLOAD_FOLDER"], profile_photo)
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            app.logger.exception("delete-photo-file")
    try:
        conn = db_connection()
        cur = conn.cursor()
        cur.execute("UPDATE users SET profile_photo=NULL WHERE id=%s", (user_id,))
        conn.commit()
        cur.close(); conn.close()
    except Exception:
        app.logger.exception("delete-photo-db")
    flash("Profile photo removed.")
    return redirect("/settings")


# ---------- ADMIN ----------
@app.route("/admin")
def admin_index():
    if not require_login() or not require_admin():
        return redirect("/login")
    try:
        # Basic cards
        stats = {"users": 0, "predictions": 0}
        conn = db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users")
        stats["users"] = int(cur.fetchone()[0])
        cur.execute("SELECT COUNT(*) FROM prediction_history")
        stats["predictions"] = int(cur.fetchone()[0])
        cur.close(); conn.close()
    except Exception:
        app.logger.exception("admin stats")
    try:
        return render_template("admin/index.html", username=session.get("username"), stats=stats)
    except Exception:
        abort(404)


@app.route("/admin/users")
def admin_users():
    if not require_login() or not require_admin():
        return redirect("/login")
    try:
        conn = db_connection()
        cur = conn.cursor(dictionary=True, buffered=True)
        cur.execute("SELECT id, username, full_name, email, is_admin, created_at FROM users ORDER BY created_at DESC")
        users = cur.fetchall()
        cur.close(); conn.close()
    except Exception:
        app.logger.exception("admin_users")
        users = []
    try:
        return render_template("admin/users.html", users=users)
    except Exception:
        abort(404)


# ---------- NEW FEATURE PAGES ----------
@app.route("/analytics")
def analytics():
    if not require_login():
        return redirect("/login")
    user_id = session.get("user_id")
    total_predictions = 0
    class_counts = {"Setosa": 0, "Versicolor": 0, "Virginica": 0}
    last_pred_class = None

    try:
        conn = db_connection()
        cur = conn.cursor(dictionary=True, buffered=True)

        cur.execute("SELECT COUNT(*) AS c FROM prediction_history WHERE user_id=%s", (user_id,))
        row = cur.fetchone()
        total_predictions = int(row["c"]) if row else 0

        cur.execute("SELECT predicted_class, COUNT(*) AS cnt FROM prediction_history WHERE user_id=%s GROUP BY predicted_class", (user_id,))
        rows = cur.fetchall()
        for r in rows:
            cls = r["predicted_class"]
            if cls in class_counts:
                class_counts[cls] = int(r["cnt"])

        cur.execute("SELECT predicted_class FROM prediction_history WHERE user_id=%s ORDER BY timestamp DESC LIMIT 1", (user_id,))
        last = cur.fetchone()
        last_pred_class = last["predicted_class"] if last else None

    except Exception:
        app.logger.exception("analytics error")
    finally:
        try:
            cur.close(); conn.close()
        except Exception:
            pass

    return render_template("analytics.html",
        total_predictions=total_predictions,
        class_counts=class_counts,
        last_pred_class=last_pred_class
    )

@app.route('/powerbi')
def powerbi_dashboard():
    return render_template('powerbi.html')



@app.route("/dataset")
def dataset():
    if not require_login():
        return redirect("/login")
    # Read iris.csv with pagination
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))
    search = (request.args.get("q") or "").strip().lower()

    csv_path = os.path.join(DATA_DIR, "iris.csv")
    if not os.path.exists(csv_path):
        return render_template("dataset.html", rows=[], page=1, pages=1, columns=[], error="Dataset file not found at data/iris.csv")

    try:
        df = pd.read_csv(csv_path)
        # Optional search across numeric-as-string and species
        if search:
            df_str = df.astype(str).apply(lambda col: col.str.lower())
            mask = df_str.apply(lambda col: col.str.contains(search, na=False))
            df = df[mask.any(axis=1)]
        total = len(df)
        pages = max(1, (total + per_page - 1) // per_page)
        start = (page - 1) * per_page
        rows = df.iloc[start:start+per_page].to_dict(orient="records")
        columns = list(df.columns)
        return render_template("dataset.html", rows=rows, page=page, pages=pages, columns=columns, q=search)
    except Exception:
        app.logger.exception("dataset read")
        return render_template("dataset.html", rows=[], page=1, pages=1, columns=[], error="Failed to read dataset")


@app.route("/model-info")
def model_info():
    if not require_login():
        return redirect("/login")
    info = {
        "loaded": model is not None and scaler is not None,
        "model_type": type(model).__name__ if model else None,
        "classes": getattr(model, "classes_", ["Setosa", "Versicolor", "Virginica"]) if model else ["Setosa", "Versicolor", "Virginica"],
        "n_estimators": getattr(model, "n_estimators", None),
        "params": getattr(model, "get_params", lambda: {})(),
        "accuracy": model_accuracy,
    }
    return render_template("model_info.html", info=info)


@app.route("/notifications")
def notifications():
    if not require_login():
        return redirect("/login")
    # Simple sample notifications in-memory for UI (DB table optional upgrade)
    feed = [
        {"type": "info", "title": "Welcome", "msg": "Thanks for joining the Iris Project!", "time": "Today"},
        {"type": "success", "title": "Model Loaded", "msg": "RandomForest model is active.", "time": "Today"} if model else {"type": "warning", "title": "Model Missing", "msg": "Upload or train model to enable predictions.", "time": "Today"},
        {"type": "info", "title": "Tips", "msg": "Use Analytics to see your prediction distribution.", "time": "This week"},
    ]
    return render_template("notifications.html", feed=feed)


@app.route("/help")
def help_page():
    return render_template("help.html")


# ---------- STATIC UPLOADS route helper ----------
@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return redirect(url_for("static", filename=f"uploads/{filename}"))


# ---------- START ----------
if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(debug=debug_mode, host=host, port=port)

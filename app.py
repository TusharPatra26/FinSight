"""FinSight Flask application. Loads ONLY the promoted model; never trains.

Run:  python app.py        (or: flask --app app run)
"""
import csv
import io
import json
import os
import re
import uuid
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

from flask import (Flask, Response, abort, flash, jsonify, redirect, render_template, request, send_from_directory,
                   url_for)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
from werkzeug.utils import secure_filename

import config
from database import db
from src import analytics
from src.analytics import CSVValidationError
from src.inference import InferenceError, InputValidationError, ModelNotFoundError, SentimentModel

BATCH_ID_RE = re.compile(r"^[0-9a-f]{32}$")
EXPORT_COLUMNS = ("input_text", "sentiment", "score_type", "score", "margin", "margin_band", "key_terms", "model_id", "created_at")
TEXT_EXPORT_COLUMNS = {"input_text", "key_terms"}
PER_PAGE = 25


def create_app(test_config=None) -> Flask:
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("FINSIGHT_SECRET_KEY", "dev-only-change-me"),
        MAX_CONTENT_LENGTH=config.MAX_UPLOAD_BYTES,
        DATABASE=str(config.DATABASE_PATH),
        UPLOAD_FOLDER=str(config.UPLOADS_DIR),
        MODEL_PATH=str(config.MODEL_PATH),
        METADATA_PATH=str(config.METADATA_PATH),
    )
    if test_config:
        app.config.update(test_config)
    db.init_db(app.config["DATABASE"])
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    state = {"model": None, "error": None}

    def get_model() -> SentimentModel:
        """Load the promoted model once; keep retrying if it was missing (e.g. experiments run later)."""
        if state["model"] is None:
            try:
                state["model"] = SentimentModel(app.config["MODEL_PATH"], app.config["METADATA_PATH"])
                state["error"] = None
            except ModelNotFoundError as exc:
                state["error"] = str(exc)
                raise
        return state["model"]

    def model_or_none():
        try:
            return get_model()
        except ModelNotFoundError:
            return None

    @lru_cache(maxsize=1)
    def dataset_info():
        return analytics.dataset_overview()

    # ------------------------------------------------------------------ security / errors
    @app.before_request
    def reject_cross_site_posts():
        """Block browser POSTs that originate from another site (simple CSRF defence for a local app)."""
        if request.method == "POST":
            origin = request.headers.get("Origin")
            if origin and urlparse(origin).netloc != request.host:
                abort(403, "Cross-site request rejected.")

    def wants_json() -> bool:
        return request.path.startswith("/api/")

    def error_response(message: str, status: int):
        if wants_json():
            return jsonify({"error": message}), status
        return render_template("error.html", message=message, status=status), status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return error_response(f"The upload is too large (limit {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB).", 413)

    @app.errorhandler(HTTPException)
    def http_error(exc):
        return error_response(exc.description or exc.name, exc.code or 500)

    @app.errorhandler(Exception)
    def unexpected(exc):
        app.logger.exception("Unhandled error")
        return error_response("Something went wrong on our side. Please try again.", 500)

    @app.errorhandler(ModelNotFoundError)
    def no_model(exc):
        return error_response(str(exc), 503)

    # ------------------------------------------------------------------ helpers
    def run_batch(file_storage, text_column):
        """Validate an uploaded CSV, predict, persist, and return everything the caller needs."""
        if file_storage is None or not file_storage.filename:
            raise CSVValidationError("No file was selected. Please choose a CSV file.")
        name = secure_filename(file_storage.filename) or "upload.csv"
        if Path(name).suffix.lower() not in config.ALLOWED_UPLOAD_EXTENSIONS:
            raise CSVValidationError("Only .csv files are accepted.")
        model = get_model()
        batch_id = uuid.uuid4().hex
        dest = Path(app.config["UPLOAD_FOLDER"]) / f"{batch_id}_{name}"
        file_storage.save(dest)
        try:
            texts, report = analytics.read_csv_texts(dest, text_column or None)
        except CSVValidationError:
            dest.unlink(missing_ok=True)   # don't keep files we rejected
            raise
        results = model.predict_batch(texts)
        db.add_results(app.config["DATABASE"], results, "batch", batch_id)
        return {"batch_id": batch_id, "filename": name, "report": report, "results": results,
                "summary": analytics.batch_summary(results)}

    def safe_cell(value):
        """Neutralise spreadsheet formula injection in exported CSVs."""
        s = "" if value is None else str(value)
        return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s

    def csv_response(rows, filename):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(EXPORT_COLUMNS)
        for r in rows:   # only free-text columns need formula protection; numbers must stay numbers (scores can be negative)
            w.writerow([safe_cell(r[c]) if c in TEXT_EXPORT_COLUMNS else r[c] for c in EXPORT_COLUMNS])
        return Response(buf.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})

    def asset(filename: str) -> str:
        """Static URL with a version taken from the file's modification time, so browsers never serve stale CSS/JS."""
        path = Path(app.static_folder) / filename
        return url_for("static", filename=filename, v=int(path.stat().st_mtime) if path.exists() else 0)

    @app.context_processor
    def inject_globals():
        m = model_or_none()
        return {"nav_model": m, "active": request.endpoint, "asset": asset}

    # ------------------------------------------------------------------ pages
    @app.route("/")
    def index():
        model = model_or_none()
        dbp = app.config["DATABASE"]
        meta = model.metadata if model else None
        return render_template("index.html", model=model, meta=meta, dataset=dataset_info(),
                               total=db.count_analyses(dbp), counts=analytics.sentiment_distribution(db.sentiment_counts(dbp)),
                               recent=db.list_analyses(dbp, limit=5), model_error=state["error"],
                               lab=analytics.lab_payload(analytics.load_experiment_results(), meta))

    @app.route("/sentiment", methods=["GET", "POST"])
    def sentiment():
        model = model_or_none()
        ctx = {"model": model, "single": None, "batch": None, "text": "", "mode": "single",
               "bands": model.bands if model else None}
        if request.method == "POST":
            f = request.files.get("file")
            try:
                if f is not None and f.filename:
                    ctx["mode"] = "batch"
                    ctx["batch"] = run_batch(f, request.form.get("text_column", "").strip())
                else:
                    ctx["text"] = request.form.get("text", "")
                    result = get_model().predict(ctx["text"])
                    db.add_results(app.config["DATABASE"], [result], "single")
                    ctx["single"] = result
            except (InputValidationError, CSVValidationError) as exc:
                ctx["mode"] = "batch" if (f is not None and f.filename) else "single"
                flash(str(exc), "error")
        return render_template("sentiment.html", **ctx)

    @app.route("/market-pulse")
    def market_pulse():
        dbp = app.config["DATABASE"]
        rows = db.list_analyses(dbp, limit=5000)
        texts, labels = [r["input_text"] for r in rows], [r["sentiment"] for r in rows]
        days = {}
        for r in db.daily_counts(dbp):
            days.setdefault(r["day"], {n: 0 for n in config.LABEL_NAMES})[r["sentiment"]] = r["n"]
        return render_template("market_pulse.html", n=len(rows), dist=analytics.sentiment_distribution(labels),
                               vocab=analytics.financial_term_counts(texts), words=analytics.top_words_by_sentiment(texts, labels),
                               days=days, dataset=dataset_info(), corpus=analytics.corpus_vocabulary_profile())

    @app.route("/model-lab")
    def model_lab():
        model = model_or_none()
        exp = analytics.load_experiment_results()
        return render_template("model_lab.html", exp=exp, model=model, lab=analytics.lab_payload(exp, model.metadata if model else None))

    @app.route("/history")
    def history():
        sentiment_f = request.args.get("sentiment") if request.args.get("sentiment") in config.LABEL_NAMES else None
        source_f = request.args.get("source") if request.args.get("source") in ("single", "batch") else None
        q = (request.args.get("q") or "").strip()[:200] or None
        try:
            page = max(1, int(request.args.get("page", 1)))
        except ValueError:
            page = 1
        dbp = app.config["DATABASE"]
        total = db.count_analyses(dbp, sentiment_f, source_f, q)
        pages = max(1, -(-total // PER_PAGE))
        page = min(page, pages)
        rows = db.list_analyses(dbp, PER_PAGE, (page - 1) * PER_PAGE, sentiment_f, source_f, q)
        return render_template("history.html", rows=rows, total=total, page=page, pages=pages,
                               filters={"sentiment": sentiment_f or "", "source": source_f or "", "q": q or ""})

    @app.route("/history/export")
    def history_export():
        rows = db.list_analyses(app.config["DATABASE"], limit=1_000_000)
        return csv_response(rows, "finsight_history.csv")

    @app.route("/history/clear", methods=["POST"])
    def history_clear():
        if request.form.get("confirm") != "yes":
            flash("History was not cleared (confirmation missing).", "error")
        else:
            n = db.clear_history(app.config["DATABASE"])
            flash(f"Cleared {n} history entries.", "ok")
        return redirect(url_for("history"))

    @app.route("/methodology")
    def methodology():
        model = model_or_none()
        meta = model.metadata if model else None
        report = None
        if config.VALIDATION_REPORT.exists():
            report = json.loads(config.VALIDATION_REPORT.read_text(encoding="utf-8"))
        return render_template("methodology.html", meta=meta, dataset=dataset_info(), report=report,
                               lab=analytics.lab_payload(analytics.load_experiment_results(), meta),
                               corpus=analytics.corpus_vocabulary_profile())

    @app.route("/analysis-figures/<path:filename>")
    def analysis_figure(filename):
        """Serve the committed EDA figures (PNG only) for the Methodology page."""
        if not filename.lower().endswith(".png"):
            abort(404)
        return send_from_directory(config.ANALYSIS_DIR / "figures", filename)

    # ------------------------------------------------------------------ API + downloads
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": 'Send JSON like {"text": "..."}'}), 400
        try:
            result = get_model().predict(data.get("text"))
        except InputValidationError as exc:
            return jsonify({"error": str(exc)}), 400
        db.add_results(app.config["DATABASE"], [result], "single")
        return jsonify(result)

    @app.route("/api/batch", methods=["POST"])
    def api_batch():
        try:
            out = run_batch(request.files.get("file"), request.form.get("text_column", "").strip())
        except (CSVValidationError, InputValidationError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"batch_id": out["batch_id"], "filename": out["filename"], "report": out["report"], "summary": out["summary"],
                        "results_preview": out["results"][:100], "download_url": url_for("download_batch", batch_id=out["batch_id"])})

    @app.route("/download/<batch_id>")
    def download_batch(batch_id):
        if not BATCH_ID_RE.match(batch_id):
            abort(404, "Unknown batch.")
        rows = db.list_analyses(app.config["DATABASE"], limit=1_000_000, batch_id=batch_id, newest_first=False)
        if not rows:
            abort(404, "Unknown batch.")
        return csv_response(rows, f"finsight_results_{batch_id[:8]}.csv")

    return app


if __name__ == "__main__":
    create_app().run(debug=False)   # `flask --app app run` also works: Flask finds create_app()

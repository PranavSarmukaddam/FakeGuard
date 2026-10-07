"""
app.py
------
Flask web application for Fake News Detection and Similar News Retrieval.

Routes:
  GET  /               → Home page
  GET  /detect         → Detection input form
  POST /detect         → Run analysis, redirect to results
  GET  /results        → Display results (reads from session)
  GET  /evaluate       → Model evaluation / metrics page
  GET  /about          → About page
  POST /api/predict    → JSON API endpoint (for AJAX / testing)
  GET  /api/status     → Returns model readiness status
  GET  /train          → Trigger training (dev convenience)

SPPU IR Syllabus:
  - End-to-end IR system demo
  - All pipeline components wired together
"""

import os
import sys
import json
import logging
import subprocess

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify, flash, send_from_directory
)

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from retrieval import analyze, models_ready, get_models
from evaluate  import load_evaluation_results, get_confusion_matrix_data, format_metrics_table, get_summary_stats
from preprocessing import preprocess_pipeline_info

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "fakeguard-platform-secret-2026")
app.config["SESSION_TYPE"] = "filesystem"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# Eager-load models on startup if they exist
if models_ready():
    try:
        get_models()
        log.info("Models loaded at startup.")
    except Exception as e:
        log.warning("Could not pre-load models: %s", e)


# ---------------------------------------------------------------------------
# Context processors
# ---------------------------------------------------------------------------
@app.context_processor
def inject_model_status():
    return {"models_trained": models_ready()}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/favicon.ico")
def favicon():
    """Serve favicon directly."""
    return send_from_directory(
        os.path.join(app.root_path, "static", "images"),
        "favicon.png",
        mimetype="image/png"
    )


@app.route("/")
def home():
    """Landing / Home page."""
    return render_template("index.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    """
    GET  → Render the detection form.
    POST → Validate input, run analysis pipeline, store results in session,
           redirect to /results.
    """
    if request.method == "GET":
        return render_template("detect.html")

    # --- POST ---
    headline = request.form.get("headline", "").strip()
    body     = request.form.get("body", "").strip()

    if not headline and not body:
        flash("Please enter at least a headline or article body.", "warning")
        return render_template("detect.html", headline=headline, body=body)

    if len(headline) + len(body) < 20:
        flash("Input is too short. Please provide more text.", "warning")
        return render_template("detect.html", headline=headline, body=body)

    if not models_ready():
        flash("Models are not trained yet. Please run the training script first.", "danger")
        return render_template("detect.html", headline=headline, body=body)

    try:
        result = analyze(headline, body, top_k=5)
        if "error" in result:
            flash(result["error"], "warning")
            return render_template("detect.html", headline=headline, body=body)

        # Store in session for results page
        session["headline"]   = headline
        session["body"]       = body[:3000]          # cap for session storage
        session["prediction"] = result["prediction"]
        session["similar"]    = result["similar"]

        return redirect(url_for("results"))

    except Exception as exc:
        log.exception("Error during analysis: %s", exc)
        flash(f"An error occurred during analysis: {exc}", "danger")
        return render_template("detect.html", headline=headline, body=body)


@app.route("/results")
def results():
    """Display classification result and similar articles."""
    headline   = session.get("headline", "")
    body       = session.get("body", "")
    prediction = session.get("prediction")
    similar    = session.get("similar", [])

    if not prediction:
        flash("No results to display. Please submit an article first.", "info")
        return redirect(url_for("detect"))

    return render_template(
        "result.html",
        headline=headline,
        body=body,
        prediction=prediction,
        similar=similar,
    )


@app.route("/evaluate")
def evaluate():
    """Model evaluation page with metrics and confusion matrix."""
    eval_data = load_evaluation_results()

    if not eval_data:
        return render_template(
            "evaluate.html",
            no_data=True,
            pipeline_info=preprocess_pipeline_info(),
        )

    results_list = eval_data.get("results", [])
    cm_data      = get_confusion_matrix_data(results_list)
    metrics_rows = format_metrics_table(results_list)
    summary      = get_summary_stats(results_list)
    pipeline_info = preprocess_pipeline_info()

    return render_template(
        "evaluate.html",
        no_data=False,
        best_model=eval_data.get("best_model"),
        trained_at=eval_data.get("trained_at"),
        metrics_rows=metrics_rows,
        cm_data=json.dumps(cm_data),
        summary=summary,
        pipeline_info=pipeline_info,
    )


@app.route("/about")
def about():
    """About page describing the project, IR concepts and team."""
    pipeline_info = preprocess_pipeline_info()
    return render_template("about.html", pipeline_info=pipeline_info)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API endpoint for programmatic access.

    Request body (JSON):
      { "headline": "...", "body": "..." }

    Response (JSON):
      { "prediction": {...}, "similar": [...] }
    """
    data = request.get_json(force=True, silent=True) or {}
    headline = str(data.get("headline", "")).strip()
    body     = str(data.get("body", "")).strip()

    if not headline and not body:
        return jsonify({"error": "headline or body is required"}), 400

    if not models_ready():
        return jsonify({"error": "Models not trained. Run train_model.py first."}), 503

    try:
        result = analyze(headline, body, top_k=5)
        return jsonify(result)
    except Exception as exc:
        log.exception("API error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/status")
def api_status():
    """Return model readiness and basic info."""
    ready = models_ready()
    info = {"models_ready": ready}
    if ready:
        eval_data = load_evaluation_results()
        info["best_model"] = eval_data.get("best_model", "unknown")
        info["trained_at"] = eval_data.get("trained_at", "unknown")
    return jsonify(info)


@app.route("/train", methods=["GET", "POST"])
def train_route():
    """
    Convenience route to trigger training from the browser (dev mode only).
    In production this should be protected or removed.
    """
    if request.method == "GET":
        return render_template("train.html")

    # POST: kick off training
    try:
        train_script = os.path.join(os.path.dirname(__file__), "src", "train_model.py")
        result = subprocess.run(
            [sys.executable, train_script],
            capture_output=True, text=True, timeout=600
        )
        stdout = result.stdout[-5000:] if result.stdout else ""
        stderr = result.stderr[-2000:] if result.stderr else ""
        success = result.returncode == 0

        if success:
            # Reload models
            try:
                get_models()
            except Exception:
                pass

        return render_template(
            "train.html",
            ran=True,
            success=success,
            stdout=stdout,
            stderr=stderr,
        )
    except subprocess.TimeoutExpired:
        return render_template(
            "train.html", ran=True, success=False,
            stdout="", stderr="Training timed out (>600 s)."
        )
    except Exception as exc:
        return render_template(
            "train.html", ran=True, success=False,
            stdout="", stderr=str(exc)
        )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page not found."), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Internal server error."), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

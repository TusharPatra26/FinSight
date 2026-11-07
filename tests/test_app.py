"""Tests for Flask routes, CSV handling, downloads and database persistence."""
import io

import pytest

from app import create_app
from database import db
from src.analytics import CSVValidationError, read_csv_texts

POS = "Operating profit rose to EUR 12.5 mn from EUR 8.1 mn in the previous year."
NEG = "The company reported a net loss and sales decreased by 15 % compared with last year."


@pytest.fixture()
def app(tmp_path):
    return create_app({"TESTING": True, "DATABASE": str(tmp_path / "test.db"), "UPLOAD_FOLDER": str(tmp_path / "uploads")})


@pytest.fixture()
def client(app):
    return app.test_client()


def csv_file(content, name="news.csv"):
    return {"file": (io.BytesIO(content if isinstance(content, bytes) else content.encode("utf-8")), name)}


# ---------------------------------------------------------------- pages
@pytest.mark.parametrize("path", ["/", "/sentiment", "/market-pulse", "/model-lab", "/history", "/methodology"])
def test_all_six_pages_render(client, path):
    assert client.get(path).status_code == 200


def test_unknown_page_is_404_with_message(client):
    r = client.get("/nope")
    assert r.status_code == 404 and b"Error 404" in r.data


def test_pages_survive_missing_model_and_api_returns_503(tmp_path):
    app = create_app({"TESTING": True, "DATABASE": str(tmp_path / "t.db"), "UPLOAD_FOLDER": str(tmp_path / "u"),
                      "MODEL_PATH": str(tmp_path / "missing.joblib")})
    c = app.test_client()
    assert c.get("/").status_code == 200 and c.get("/sentiment").status_code == 200
    r = c.post("/api/predict", json={"text": POS})
    assert r.status_code == 503 and "run_experiments" in r.get_json()["error"]


# ---------------------------------------------------------------- single prediction API + persistence
def test_api_predict_returns_expected_format_and_persists(client, app):
    r = client.post("/api/predict", json={"text": POS})
    assert r.status_code == 200
    body = r.get_json()
    assert body["sentiment"] == "positive" and body["score_type"] == "decision_score" and body["margin_band"] in ("Low", "Medium", "High")
    rows = db.list_analyses(app.config["DATABASE"])
    assert len(rows) == 1
    row = rows[0]
    assert row["input_text"] == POS and row["sentiment"] == "positive" and row["source"] == "single"
    import re
    assert re.fullmatch(r"LinearSVC-[0-9a-f]{8}", row["model_id"]) and row["created_at"] and isinstance(row["score"], float)


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": 5}, {"text": None}, {"text": "x" * 2001}])
def test_api_predict_rejects_bad_text(client, app, payload):
    r = client.post("/api/predict", json=payload)
    assert r.status_code == 400 and "error" in r.get_json()
    assert db.count_analyses(app.config["DATABASE"]) == 0   # nothing stored for failed requests


def test_api_predict_rejects_non_json(client):
    assert client.post("/api/predict", data="hello", content_type="text/plain").status_code == 400
    assert client.post("/api/predict", data="{bad json", content_type="application/json").status_code == 400


def test_html_form_single_text_flow_and_missing_input(client, app):
    r = client.post("/sentiment", data={"text": NEG})
    assert r.status_code == 200 and b"negative" in r.data and b"not a probability" in r.data
    r = client.post("/sentiment", data={"text": "  "})
    assert r.status_code == 200 and b"The text is empty" in r.data
    assert db.count_analyses(app.config["DATABASE"]) == 1


def test_user_text_is_html_escaped(client):
    r = client.post("/sentiment", data={"text": "<script>alert(1)</script> profit rose"})
    assert b"<script>alert(1)</script>" not in r.data


# ---------------------------------------------------------------- CSV batch
GOOD_CSV = f"id,text\n1,{POS}\n2,{NEG}\n3,\n4,The annual general meeting will be held in Helsinki.\n"


def test_batch_upload_end_to_end_with_download_and_history(client, app):
    r = client.post("/api/batch", data=csv_file(GOOD_CSV), content_type="multipart/form-data")
    assert r.status_code == 200
    body = r.get_json()
    assert body["report"]["rows_analysed"] == 3 and body["report"]["skipped_empty"] == 1 and body["report"]["text_column"] == "text"
    assert body["summary"]["n"] == 3 and len(body["results_preview"]) == 3
    assert [x["sentiment"] for x in body["results_preview"]][:2] == ["positive", "negative"]
    dl = client.get(body["download_url"])
    assert dl.status_code == 200 and dl.mimetype == "text/csv" and "attachment" in dl.headers["Content-Disposition"]
    lines = dl.get_data(as_text=True).strip().splitlines()
    assert lines[0].startswith("input_text,sentiment,score_type") and len(lines) == 4
    assert db.count_analyses(app.config["DATABASE"], source="batch", batch_id=body["batch_id"]) == 3


def test_batch_via_html_form_shows_summary_and_download_link(client):
    r = client.post("/sentiment", data=csv_file(GOOD_CSV), content_type="multipart/form-data")
    assert r.status_code == 200 and b"Download results CSV" in r.data
    assert b"Rows analysed" in r.data and b"Skipped" in r.data and b'data-chart="doughnut"' in r.data


def test_single_column_csv_and_explicit_column_choice(client):
    r = client.post("/api/batch", data=csv_file(f"headline\n{POS}\n"), content_type="multipart/form-data")
    assert r.status_code == 200 and r.get_json()["report"]["text_column"] == "headline"
    data = {**csv_file(f"a,b\nx,{POS}\n"), "text_column": "B"}
    assert client.post("/api/batch", data=data, content_type="multipart/form-data").status_code == 200


@pytest.mark.parametrize("files,msg", [
    ({}, "No file"),
    (csv_file("", "empty.csv"), "empty"),
    (csv_file("text\n", "header_only.csv"), "no data rows"),
    (csv_file("id,other\n1,2\n", "nocol.csv"), "which column"),
    (csv_file("text\n,\n", "blank.csv"), "No usable text"),
    (csv_file(POS, "x.txt"), "Only .csv"),
    (csv_file(POS, "malware.exe"), "Only .csv"),
])
def test_invalid_uploads_are_rejected_with_message(client, app, files, msg):
    r = client.post("/api/batch", data=files, content_type="multipart/form-data")
    assert r.status_code == 400 and msg.lower() in r.get_json()["error"].lower()
    assert db.count_analyses(app.config["DATABASE"]) == 0


def test_unknown_text_column_is_reported(client):
    data = {**csv_file(f"text\n{POS}\n"), "text_column": "nope"}
    r = client.post("/api/batch", data=data, content_type="multipart/form-data")
    assert r.status_code == 400 and "nope" in r.get_json()["error"]


def test_too_many_rows_rejected_not_truncated(tmp_path):
    p = tmp_path / "big.csv"
    p.write_text("text\n" + "profit rose\n" * 11, encoding="utf-8")
    with pytest.raises(CSVValidationError, match="limit"):
        read_csv_texts(p, max_rows=10)


def test_oversized_upload_returns_413(tmp_path):
    app = create_app({"TESTING": True, "DATABASE": str(tmp_path / "t.db"), "UPLOAD_FOLDER": str(tmp_path / "u"), "MAX_CONTENT_LENGTH": 200})
    r = app.test_client().post("/api/batch", data=csv_file("text\n" + "profit rose\n" * 100), content_type="multipart/form-data")
    assert r.status_code == 413 and "too large" in r.get_json()["error"]


def test_unsafe_filename_is_sanitised_and_rejected_file_not_kept(client, app, tmp_path):
    client.post("/api/batch", data=csv_file(f"text\n{POS}\n", "../../evil name.csv"), content_type="multipart/form-data")
    saved = list((tmp_path / "uploads").iterdir())
    assert len(saved) == 1 and ".." not in saved[0].name and "/" not in saved[0].name
    client.post("/api/batch", data=csv_file("id,other\n1,2\n", "bad.csv"), content_type="multipart/form-data")
    assert len(list((tmp_path / "uploads").iterdir())) == 1   # the rejected file was removed


def test_latin1_encoded_csv_is_accepted(client):
    content = "text\nProfit rose at Sepp\xe4l\xe4 by 5 %\n".encode("latin-1")
    r = client.post("/api/batch", data=csv_file(content), content_type="multipart/form-data")
    assert r.status_code == 200


def test_download_neutralises_spreadsheet_formulas(client):
    r = client.post("/api/batch", data=csv_file('text\n"=SUM(1,1) profit rose"\n'), content_type="multipart/form-data")
    body = client.get(r.get_json()["download_url"]).get_data(as_text=True)
    assert "'=SUM" in body


def test_download_keeps_negative_scores_numeric(client, app):
    """Regression: formula-injection protection must not prefix numbers like -0.02 (a demo run exposed this)."""
    batch = "b" * 32
    result = {"text": "Weak signal text", "sentiment": "positive", "score": -0.0197, "score_type": "decision_score", "margin": 0.1,
              "margin_band": "Low", "explanation": {"supporting": []}, "model_id": "M@1"}
    db.add_results(app.config["DATABASE"], [result], "batch", batch)
    body = client.get(f"/download/{batch}").get_data(as_text=True)
    assert ",-0.0197," in body and "'-0.0197" not in body


def test_download_rejects_bad_or_unknown_batch_ids(client):
    assert client.get("/download/not-a-valid-id").status_code == 404
    assert client.get("/download/" + "a" * 32).status_code == 404


# ---------------------------------------------------------------- history
def test_history_filters_search_pagination_and_export(client, app):
    client.post("/api/predict", json={"text": POS})
    client.post("/api/predict", json={"text": NEG})
    assert b"2 entries" in client.get("/history").data
    assert b"1 entry" in client.get("/history?sentiment=negative").data
    assert b"1 entry" in client.get("/history?q=net+loss").data
    assert b"2 entries" in client.get("/history?sentiment=bogus&page=abc").data    # bad filters are ignored
    exp = client.get("/history/export")
    assert exp.mimetype == "text/csv" and len(exp.get_data(as_text=True).strip().splitlines()) == 3


def test_history_search_treats_wildcards_literally(client, app):
    client.post("/api/predict", json={"text": POS})
    assert b"0 entries" in client.get("/history?q=%25").data


def test_history_clear_requires_confirmation(client, app):
    client.post("/api/predict", json={"text": POS})
    client.post("/history/clear", data={})
    assert db.count_analyses(app.config["DATABASE"]) == 1
    client.post("/history/clear", data={"confirm": "yes"})
    assert db.count_analyses(app.config["DATABASE"]) == 0


def test_cross_site_post_is_rejected(client):
    r = client.post("/api/predict", json={"text": POS}, headers={"Origin": "http://evil.example"})
    assert r.status_code == 403


def test_dashboard_and_pulse_reflect_stored_data(client):
    client.post("/api/predict", json={"text": POS})
    assert b'data-count="1">1</span>' in client.get("/").data          # "Analyses stored" KPI
    pulse = client.get("/market-pulse").data
    assert b"Revenue \\u0026 profitability" in pulse and b'data-count="1">1</span>' in pulse and b"Texts analysed" in pulse


def test_empty_states_render_helpfully(client):
    assert b"No analyses yet" in client.get("/market-pulse").data
    assert b"No analyses yet" in client.get("/history").data
    assert b"Nothing analysed yet" in client.get("/").data


def test_dashboard_kpis_come_from_the_real_metadata(client):
    import json, config
    meta = json.loads(config.METADATA_PATH.read_text(encoding="utf-8"))
    page = client.get("/").data.decode()
    assert f'data-count="{meta["final_test_metrics"]["macro_f1"]}"' in page
    assert f'data-count="{meta["final_test_metrics"]["accuracy"]}"' in page


def test_model_lab_numbers_and_confusion_data_come_from_experiment_outputs(client):
    import json, config
    metrics = json.loads((config.RESULTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    page = client.get("/model-lab").data.decode()
    assert f'{metrics["production"]["test_metrics"]["macro_f1"]:.3f}' in page
    assert 'id="cm-data"' in page and '"Production"' in page and str(metrics["production"]["confusion_matrix_test"][0][0]) in page
    assert page.count("data-chart=") >= 5          # comparison, speed and reliability charts are present


@pytest.mark.parametrize("path", ["/", "/sentiment", "/market-pulse", "/model-lab", "/history", "/methodology"])
def test_pages_are_responsive_accessible_and_offline(client, path):
    import re
    page = client.get(path).data.decode()
    assert 'name="viewport"' in page and '<html lang="en">' in page and 'id="main"' in page
    # Nothing may be LOADED from the internet (scripts, stylesheets, images); outbound <a> links are allowed but must be safe.
    assert not re.findall(r'<(?:script|img|link)\b[^>]*\b(?:src|href)="https?://', page), "pages must not load external resources"
    for tag in re.findall(r'<a\b[^>]*href="https?://[^>]*>', page):
        assert 'rel="noopener noreferrer"' in tag, tag
    assert "chart.umd.js" in page and "/static/css/style.css" in page


def test_static_assets_and_figures_are_served(client):
    for p in ("/static/css/style.css", "/static/js/main.js", "/static/js/charts.js", "/static/vendor/chart.umd.js",
              "/static/fonts/ibm-plex-serif-latin-500-normal.woff2", "/static/fonts/ibm-plex-sans-latin-400-normal.woff2",
              "/analysis-figures/B_class_distribution.png"):
        assert client.get(p).status_code == 200, p
    assert client.get("/analysis-figures/eda.ipynb").status_code == 404
    assert client.get("/analysis-figures/..%2Fconfig.py").status_code == 404
    assert client.get("/analysis-figures/missing.png").status_code == 404


def test_static_urls_are_versioned_so_browsers_never_serve_stale_assets(client):
    import re
    page = client.get("/").data.decode()
    for name in ("css/style.css", "js/charts.js", "js/main.js", "vendor/chart.umd.js"):
        assert re.search(rf"/static/{re.escape(name)}\?v=\d+", page), name


def test_charts_js_handles_missing_options_attribute():
    """Regression: JSON.parse(null) returns null, which crashed every chart without data-options."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / "static" / "js" / "charts.js").read_text(encoding="utf-8")
    assert "v == null ? fallback : v" in src


def test_single_result_page_shows_scores_band_and_words(client):
    page = client.post("/sentiment", data={"text": NEG}).data.decode()
    assert "not probabilities" in page and "badge--" in page and "Influential words" in page and "decreased" in page
    assert "correct" in page and "% of the time" in page


def test_ibm_plex_is_bundled_and_scoped_to_every_page_except_the_overview(client):
    from pathlib import Path
    fonts = Path(__file__).resolve().parent.parent / "static" / "fonts"
    woff = sorted(p.name for p in fonts.glob("*.woff2"))
    assert len(woff) == 8 and sum(n.startswith("ibm-plex-serif") for n in woff) == 3 and sum(n.startswith("ibm-plex-sans") for n in woff) == 5
    assert all(p.read_bytes()[:4] == b"wOF2" for p in fonts.glob("*.woff2"))             # genuine woff2 files
    assert (fonts / "OFL-IBM-Plex-Sans.txt").exists() and (fonts / "OFL-IBM-Plex-Serif.txt").exists()   # licences travel with the fonts
    css = client.get("/static/css/style.css").data.decode()
    assert css.count("@font-face") == 8 and 'font-family: "IBM Plex Serif"' in css and 'font-family: "IBM Plex Sans"' in css
    assert "body:not(.page-index)" in css and "font-display: swap" in css
    assert client.get("/static/fonts/ibm-plex-sans-latin-400-normal.woff2").mimetype == "font/woff2"
    assert 'class="page-index"' in client.get("/").data.decode()                           # the Overview keeps its original type
    for path, cls in (("/sentiment", "page-sentiment"), ("/market-pulse", "page-market_pulse"), ("/model-lab", "page-model_lab"),
                      ("/history", "page-history"), ("/methodology", "page-methodology"), ("/nope", "page-other")):
        assert f'class="{cls}"' in client.get(path).data.decode(), path


def test_footer_is_structured_informative_and_dateless(client):
    import re
    page = client.get("/history").data.decode()
    footer = page[page.index("<footer"):page.index("</footer>")]
    for text in ("Explore", "Research", "Data &amp; credits", "Production model", "LinearSVC", "test macro-F1", "not investment advice",
                 "Runs locally", "Back to top", "Financial PhraseBank", "CC BY-NC-SA 3.0", "IBM Plex"):
        assert text in footer, text
    for endpoint in ("/", "/sentiment", "/market-pulse", "/model-lab", "/history", "/methodology#limits", "/methodology#dataset"):
        assert f'href="{endpoint}"' in footer, endpoint
    assert not re.search(r"\b(?:19|20)\d\d\b", footer), "footer must not contain any year or date"
    assert "&copy;" not in footer and "©" not in footer


def test_no_project_dates_in_published_files():
    """The project date must not appear in the UI, metadata, notebooks, results or documentation."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    files = [root / "README.md", root / "app.py", root / "config.py", root / "models" / "metadata.json", root / "experiments" / "results" / "metrics.json",
             root / "experiments" / "results" / "comparison.csv", root / "static" / "css" / "style.css"]
    for pattern in ("templates/*.html", "src/*.py", "database/*.py", "experiments/*.py", "analysis/*.ipynb", "static/js/*.js"):
        files += sorted(root.glob(pattern))
    iso = re.compile(r"\b(?:19|20)\d\d-\d\d-\d\d\b")
    offenders = {f.relative_to(root).as_posix(): iso.findall(f.read_text(encoding="utf-8"))[:2] for f in files if iso.search(f.read_text(encoding="utf-8"))}
    assert not offenders, offenders


def test_init_db_strips_legacy_dated_model_ids(tmp_path):
    from database import db
    path = tmp_path / "legacy.db"
    db.init_db(path)
    with db.connect(path) as conn:
        for mid in ("LinearSVC@2000-01-01", "LinearSVC-3f9a1c2e", "odd@name"):   # old "Name@YYYY-MM-DD" format (placeholder date)
            conn.execute("INSERT INTO analyses (source, input_text, sentiment, score, score_type, model_id, created_at) VALUES ('single', 'x', 'neutral', 0.1, 'decision_score', ?, 't')", (mid,))
    db.init_db(path)                       # runs the migration
    db.init_db(path)                       # and is idempotent
    ids = sorted(r["model_id"] for r in db.list_analyses(path))
    assert ids == ["LinearSVC", "LinearSVC-3f9a1c2e", "odd@name"]


def test_navigation_marks_the_current_page(client):
    assert b'aria-current="page"' in client.get("/history").data


def test_model_lab_shows_real_experiment_results(client):
    page = client.get("/model-lab").data
    for name in (b"LinearSVC", b"ComplementNB", b"DummyClassifier", b"SGDClassifier"):
        assert name in page


# ---------------------------------------------------------------- database layer directly
def test_db_roundtrip_and_sql_injection_safety(tmp_path):
    path = tmp_path / "x.db"
    db.init_db(path)
    evil = "'); DROP TABLE analyses; --"
    result = {"text": evil, "sentiment": "neutral", "score": 0.5, "score_type": "decision_score", "margin": 1.0, "margin_band": "High",
              "explanation": {"supporting": [{"term": "a", "weight": 1}]}, "model_id": "M@1"}
    assert db.add_results(path, [result], "single") == 1
    assert db.list_analyses(path)[0]["input_text"] == evil
    assert db.sentiment_counts(path) == {"neutral": 1}
    with pytest.raises(Exception):
        db.add_results(path, [result], "bogus-source")

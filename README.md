# FinSight

**Financial news sentiment analytics, built as a reproducible machine-learning engineering project.**

FinSight classifies sentences from financial news as **positive, negative or neutral**, explains *which words* drove each decision, and reports honestly *how well the model works* on data it has never seen. Everything runs locally: no paid APIs, no LLM calls, no market-data feeds, no cloud inference.

> FinSight is a financial-**text** NLP research platform. It is **not** a trading tool: it uses no prices or market data, forecasts nothing, and gives no investment advice.

---

## 1. Problem statement

Given one sentence of financial news, predict its tone. This is harder than ordinary sentiment analysis:

- **Context flips meaning.** "Loss narrowed" is good news; "profit fell" is bad news; both mention money.
- **Labels are subjective.** In the dataset used here, annotators disagreed on about half the sentences.
- **Classes are imbalanced.** About 59% of sentences are neutral, so accuracy alone is misleading.

The project's goals are the engineering discipline around the model: a thorough exploratory analysis before modelling, one fair and centralised model comparison, leak-free evaluation, honest reporting, and a usable application on top.

## 2. Features

- Single-sentence analysis and **CSV batch analysis** (validated uploads, downloadable results)
- Predicted sentiment with **model scores clearly labelled as scores** (not probabilities) plus a **Low/Medium/High reliability band** whose accuracy was measured on validation data
- **Influential words** for each prediction, with the limitations of such explanations stated
- **Market Pulse**: sentiment mix, daily activity and financial-vocabulary analytics of your analyses, plus a reference-corpus profile
- **Model Laboratory**: leaderboard, charts, interactive confusion matrices, hyperparameter results. Read live from `experiments/results/`, never hard-coded
- **Analysis history** in SQLite: search, filter, paginate, export, clear
- **Methodology** page and a full EDA notebook and report
- Responsive "quantitative terminal meets research notebook" interface that respects reduced-motion settings

## 3. Technology stack

| Layer | Tools |
|---|---|
| Backend | Python 3.11+ (tested on 3.12 and 3.13), Flask, Jinja2, SQLite |
| Data science | pandas, NumPy, SciPy, Matplotlib, Seaborn |
| Machine learning | scikit-learn (Pipelines, TF-IDF), joblib |
| NLP | NLTK (stopword list for EDA), TF-IDF n-grams |
| Frontend | HTML5, CSS3, vanilla JavaScript, Chart.js 4.4.1 (bundled locally, MIT), IBM Plex Serif and IBM Plex Sans (bundled locally in `static/fonts/`, SIL Open Font License 1.1; the Overview page keeps its own type, every other page uses IBM Plex) |
| Testing | pytest, pytest-cov |

Deliberately absent: FastAPI, React/Next.js, any external API, any LLM service.

## 4. Architecture

```
 Hugging Face Hub                                       browser
        |  (one-off download)                              ^
        v                                                  |
 src/data.py ---> data/raw  ---> data/processed            v
        |                                           app.py  (Flask routes)
        |  clean() / get_splits()  (seeded, 70/15/15)   |       |        |
        v                                               v       v        v
 analysis/eda.ipynb      experiments/run_experiments.py   src/    src/    database/
 (EDA + report)            |  model_registry.py          infer-  analyt-  db.py
                           |  evaluation.py              ence.py ics.py     |
                           v                               ^                v
              experiments/results/*  -----(Model Lab)------+             SQLite
              models/best_model.joblib + metadata.json --(only the promoted model is loaded)
                                src/preprocessing.py  (shared by training AND inference)
```

Key design rules: one registry for all candidate models; one preprocessing module shared by training and inference; the app **never trains** and loads only the promoted model.

## 5. Dataset, source and attribution

**Financial PhraseBank v1.0**, 50%-agreement subset (`sentences_50agree`): 4,846 English financial-news sentences labelled negative / neutral / positive by finance-literate annotators.

- Source: <https://huggingface.co/datasets/takala/financial_phrasebank> (the authors' original archive, `FinancialPhraseBank-v1.0.zip`)
- **Licence: Creative Commons Attribution-NonCommercial-ShareAlike 3.0. Non-commercial use only.** Commercial users should contact the dataset authors.
- Citation: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* Journal of the Association for Information Science and Technology, 65.

> **Why not `load_dataset(...)`?** Current versions of the `datasets` library refuse script-based datasets, and `takala/financial_phrasebank` is one. `src/data.py` therefore downloads the authors' original zip from the same Hub repository with `huggingface_hub`, reads only `Sentences_50Agree.txt`, and saves it byte-for-byte to `data/raw/`. The other agreement levels are never mixed in.

Cleaning (documented in the EDA, applied by `src.data.clean()`): sentences with **conflicting labels are dropped** (2 sentences), and **exact same-label duplicates are collapsed** (6 copies), giving **4,836 sentences**.

## 6. EDA methodology and major findings

The EDA ([`analysis/eda.ipynb`](analysis/eda.ipynb), summarised in [`analysis/eda_report.md`](analysis/eda_report.md)) has ten sections covering data integrity, class balance, length, vocabulary, class-wise language, a hand-written financial lexicon, data quality, annotator agreement, and feature representation. Each section states what was investigated, what the data showed, why it matters, and what it implies for modelling. Main findings (all produced by the notebook):

| Finding | Evidence |
|---|---|
| The data is imbalanced | neutral 59.4%, positive 28.1%, negative 12.5% (4.77 : 1). An always-"neutral" model scores **0.594 accuracy but only 0.248 macro-F1** |
| Length is not a signal | mean 23.1 words; class differences are statistically detectable but tiny (effect size 0.013) |
| Numbers are a signal | 75% of negative sentences contain digits vs 47% of neutral ones |
| Direction words carry sentiment; money words are ambiguous | risk terms are 76% negative, growth terms 76% positive, but "loss" is 46% negative and 46% positive ("loss narrowed") |
| Language is sparse | 9,077 word types, 49% of which occur once |
| Labels are genuinely uncertain | a quick check scored macro-F1 0.59 on non-unanimous sentences vs 0.88 on unanimous ones |
| Stopword removal hurts | standard lists delete "not", "down", "against"; quick check: macro-F1 0.730 to 0.644 |
| Data quirks | 8 repeated sentences (2 with conflicting labels), 91 rows with encoding damage in the source file (`Seppälä` shows as `Sepp+ñl+ñ`) |

Limitations of the EDA: word clouds were deliberately omitted (the quantitative charts cover the same ground); the financial lexicon is a small manual list, not a complete dictionary; association scores are not causal. `eda_report.md` is a written summary of the notebook's findings; the notebook is the source of truth.

## 7. ML experimentation methodology

All modelling lives in one place, [`experiments/`](experiments/):

- `model_registry.py`: the six candidates and their small tuning grids. Adding or removing a model is one entry.
- `evaluation.py`: every metric, timing, confusion matrix and comparison chart.
- `run_experiments.py`: the whole experiment, in an order that protects the test set.

**Design:** one stratified **70 / 15 / 15** split (3,384 / 726 / 726 sentences) with a fixed seed (42), identical for every model; TF-IDF is fitted **inside** each scikit-learn Pipeline so nothing leaks; 5-fold cross-validation on the training split; hyperparameters and the winner chosen on **validation macro-F1**; the **test set is untouched until selection is finished**, then the winner is refit on train + validation and scored once.

## 8. Candidate models

| # | Model | Grid searched (plus unigram vs unigram+bigram for all) |
|---|---|---|
| 1 | DummyClassifier (majority class) | none, baseline |
| 2 | Multinomial Naive Bayes | alpha |
| 3 | Complement Naive Bayes | alpha |
| 4 | Logistic Regression | C, class_weight |
| 5 | Linear SVC | C, class_weight |
| 6 | SGDClassifier (log-loss) | alpha, class_weight |

## 9. Model-selection criteria

1. **Primary metric: validation macro-F1.** Macro-F1 weights all three classes equally, so it cannot be inflated by predicting the majority class. Accuracy is reported but never decides.
2. **Tie-breaks:** cross-validation macro-F1, then faster inference, then model name.
3. Test results are reported for every candidate but **never influence selection**.

## 10. Actual evaluation results

All numbers are from `experiments/results/` (generated by `python -m experiments.run_experiments`). Test columns come from each model fitted on the training split only and were computed after selection.

| Model | CV macro-F1 (train, 5-fold) | Val macro-F1 | Test accuracy | Test macro-F1 | Test weighted-F1 |
|---|---|---|---|---|---|
| **LinearSVC (selected)** | 0.720 ± 0.011 | **0.762** | 0.774 | 0.731 | 0.769 |
| SGDClassifier | 0.709 ± 0.020 | 0.747 | 0.770 | 0.725 | 0.767 |
| LogisticRegression | 0.717 ± 0.012 | 0.743 | 0.773 | 0.735 | 0.771 |
| ComplementNB | 0.685 ± 0.019 | 0.690 | 0.729 | 0.678 | 0.725 |
| MultinomialNB | 0.660 ± 0.029 | 0.683 | 0.737 | 0.673 | 0.724 |
| DummyClassifier | 0.248 ± 0.000 | 0.248 | 0.594 | 0.248 | 0.442 |

**Production model** (LinearSVC, C=1, balanced class weights, unigrams + bigrams; refit on train + validation, scored once on the held-out test set):

| Accuracy | Macro-F1 | Weighted-F1 | Macro precision | Macro recall |
|---|---|---|---|---|
| **0.781** | **0.739** | 0.777 | 0.752 | 0.730 |

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| negative | 0.700 | 0.692 | 0.696 | 91 |
| neutral | 0.809 | 0.875 | 0.841 | 431 |
| positive | 0.747 | 0.623 | 0.679 | 204 |

**How to read this honestly:**

- Every real model clearly beats the baseline, and the linear models beat Naive Bayes by about 5 macro-F1 points.
- **The top three (LinearSVC, SGD, Logistic Regression) are statistically close**: validation macro-F1 within 1.9 points, overlapping cross-validation scores, and on the test set Logistic Regression scored marginally higher. Selection followed the pre-declared rule and was not revised after seeing test results.
- The winner's validation score (0.762) is slightly optimistic, because settings and model were both chosen on validation. The test numbers are the honest estimate.
- The weakest spot is **positive recall (0.62)**: positive sentences are most often predicted neutral (69 of 204), echoing the EDA finding that the neutral/polar boundary is the most subjective.
- **Confidence:** LinearSVC outputs scores, not probabilities, so FinSight shows scores labelled as such plus a reliability band (margin between the top two class scores). Measured out-of-fold on train + validation, predictions in the Low / Medium / High bands were correct **58% / 81% / 94%** of the time. This is an empirical reliability indicator, not a calibrated probability.

[`analysis/model_experiments.ipynb`](analysis/model_experiments.ipynb) re-fits every candidate live and asserts the numbers match the saved results.

## 11. Installation

Requires Python 3.11 or newer.

```powershell
# Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Keep the virtual environment activated for every command below.** In particular, Jupyter's default `python3` kernel starts whichever `python` is first on your `PATH`; without the environment activated, notebooks may run under a different Python and fail with `ModuleNotFoundError`.

## 12. Dataset download

The repository already contains the raw and processed data, so this step is only needed to re-create it. It needs internet access once.

```bash
python -m src.data --force
```

This downloads the authors' archive from the Hugging Face Hub, writes `data/raw/Sentences_50Agree.txt` unchanged, validates it, and writes `data/processed/phrasebank_50agree.csv` and `data/processed/validation_report.json`. Without `--force` it reuses an existing raw file.

## 13. Running the EDA

```bash
jupyter lab analysis/eda.ipynb
```

or execute it headlessly (this overwrites the notebook's saved outputs and `analysis/figures/`):

```bash
python -m jupyter nbconvert --to notebook --execute --inplace analysis/eda.ipynb
```

The first run downloads NLTK's small English stopword list automatically (internet needed once). Section H also reads the other agreement levels from the cached dataset archive. Re-running the notebook reproduces every printed number and all seven figures exactly; the written interpretations in the notebook are static text that was authored from those results.

If a notebook fails with `ModuleNotFoundError`, Jupyter picked a kernel from another Python: activate the virtual environment first, or register this one with `python -m ipykernel install --user --name finsight` and select it.

## 14. Training and selecting models

```bash
python -m experiments.run_experiments
```

Takes about one minute. It tunes all six candidates, selects the winner, writes `experiments/results/` (`comparison.csv`, `metrics.json`, `comparison.png`, `confusion_matrices/`), then **promotes** the winner to `models/best_model.joblib` and `models/metadata.json`. Only the promoted model is saved. The run is deterministic: the same data and library versions give the same metrics (only timings change). Optionally re-verify with `analysis/model_experiments.ipynb`.

## 15. Running the web application

```bash
python app.py
```

Then open <http://127.0.0.1:5000>. (Alternatively `flask --app app run --port 5000`.) The app loads only `models/best_model.joblib`; it never retrains. If that file is missing, pages show a clear message and the API returns HTTP 503 until you run the experiments.

- Routes: `/` overview, `/sentiment`, `/market-pulse`, `/model-lab`, `/history`, `/methodology`
- API: `POST /api/predict` with `{"text": "..."}`; `POST /api/batch` with a multipart CSV `file`
- CSV format: a header row and a text column named `text`, `sentence`, `headline` or `title` (or choose the column); up to 2 MB and 5,000 rows
- History is stored in `database/finsight.db` (created on first run; git-ignored). Uploaded CSVs are kept in `uploads/` (git-ignored)
- Set `FINSIGHT_SECRET_KEY` in the environment to override the development session key

## 16. Running the tests

```bash
python -m pytest -q
```

With coverage:

```bash
python -m pytest -q --cov=src --cov=database --cov=experiments --cov=app --cov=config --cov-report=term-missing
```

108 tests (about 88% line coverage of the Python code; the remainder is mostly the experiment runner's `main()`, which is exercised by running it, and the network download). The suite covers dataset loading and validation, splits, the model registry, metric calculations (against hand-computed values), the promoted artifact and its metadata, single and batch inference, explanation structure, CSV validation, uploads (empty, malformed, oversized, wrong type, unsafe names), downloads, history persistence, security checks and all routes.

## 17. Reproducibility workflow

```bash
# 1. Install dependencies
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# 2. Download and validate the dataset
python -m src.data --force
# 3. Run the EDA
python -m jupyter nbconvert --to notebook --execute --inplace analysis/eda.ipynb
# 4 + 5. Run the experiments and promote the best model
python -m experiments.run_experiments
# 6. Launch Flask
python app.py
# 7. Run the tests (in another terminal)
python -m pytest -q
```

**Verified.** This workflow was run end to end from an export of the repository in a brand-new virtual environment (Windows 11, Python 3.13, packages installed from `requirements.txt`): the forced dataset re-download reproduced the raw and processed files byte-for-byte; the EDA notebook ran without errors and gave identical output on two consecutive runs; the experiments reproduced every metric, the winning model's predictions and all confusion-matrix images exactly (only timings differ); the app served all six pages, single and CSV analysis, and downloads; and all 108 tests passed.

## 18. Limitations

- **Narrow data.** About 4,800 sentences from one annotated collection of mostly Nordic company news, in English, from the 2000s. Other domains, regions or writing styles may behave differently.
- **Subjective labels.** Annotators disagreed on many sentences, which caps achievable accuracy. Only aggregated labels are published, so inter-annotator reliability (for example Cohen's kappa) is **not** computed.
- **No context.** Each sentence is judged alone, without company, date or market expectations.
- **Neutral vs positive is hard.** Positive sentences are most often mistaken for neutral.
- **Bag-of-words ceiling.** TF-IDF with two-word phrases only partly captures negation and word order.
- **Close model ranking.** The three best linear models are not meaningfully separable on this data size.
- **Margin bands are approximate.** They were measured with models trained on 80% of train + validation, and the production model is trained on all of it.
- **Word explanations show association, not causation**, and are for the predicted class only.
- **Not implemented:** word clouds and spaCy named-entity recognition (both optional in the brief).
- **Local, single-user tool.** It has basic safeguards (upload validation, escaped output, parameterised SQL, formula-safe CSV export, cross-site POST rejection), but it is not hardened for exposure to the internet. Do not deploy it publicly as-is.
- **Licence:** the dataset is non-commercial (CC BY-NC-SA 3.0); do not use this project commercially without contacting the dataset authors.
- Developed and tested on Windows 11 with Python 3.12 and 3.13.

## 19. Future improvements

- Compare against contextual models (for example a fine-tuned FinBERT) as a clearly separated, optional experiment
- Calibrate scores properly (for example cross-validated sigmoid/isotonic calibration) and evaluate calibration
- Bootstrap confidence intervals or repeated cross-validation, so close models can be compared with uncertainty
- Add the other agreement levels as a robustness analysis and study the contested sentences directly
- Add named-entity recognition to group sentiment by company
- Add a "difficult sentences" view for low-margin predictions to support human review

---

## Project structure

```
FinSight/
├── app.py                  Flask application factory and routes
├── config.py               paths, limits, seed (single source of configuration)
├── requirements.txt
├── data/
│   ├── raw/                original file, untouched
│   └── processed/          cleaned CSV + validation report
├── analysis/
│   ├── eda.ipynb           exploratory data analysis (executed)
│   ├── eda_report.md       written summary of the EDA
│   ├── model_experiments.ipynb   presents and re-verifies the experiment results
│   └── figures/            EDA figures
├── experiments/
│   ├── model_registry.py   candidate models and grids
│   ├── evaluation.py       metrics, timing, confusion matrices, comparison charts
│   ├── run_experiments.py  the centralised experiment + promotion
│   └── results/            comparison.csv, metrics.json, comparison.png, confusion_matrices/
├── models/                 best_model.joblib + metadata.json (promoted model only)
├── src/
│   ├── data.py             dataset download, validation, cleaning, splits
│   ├── preprocessing.py    shared text normalisation
│   ├── inference.py        model loading, prediction, explanations, input validation
│   └── analytics.py        CSV validation and aggregation / NLP analytics
├── database/db.py          SQLite history layer
├── templates/              Jinja2 pages (base, index, sentiment, market_pulse, model_lab, history, methodology, error)
├── static/                 css/style.css, js/main.js, js/charts.js, vendor/chart.umd.js
├── tests/                  test_models.py, test_inference.py, test_app.py
└── uploads/                uploaded CSVs (git-ignored)
```

Small additions to the original plan, each with a reason: `src/data.py` (dataset loading is reused by the EDA, experiments and tests), `analysis/figures/` (EDA plots), `analysis/model_experiments.ipynb` (requested presentation of experiment results), `templates/error.html` (error pages), `static/vendor/` (Chart.js bundled so the app works offline).

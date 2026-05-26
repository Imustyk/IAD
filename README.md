# 📊 Data Science SaaS

An end-to-end data analytics web application built with **Streamlit**,
**scikit-learn** and **Plotly**, designed for the master's-level
**"Data Analysis Tools"** course.

The application implements the full course workflow:

| # | Course requirement | Where in the app |
|---|--------------------|------------------|
| 1 | Business-case formulation | Home page |
| 2 | Identification of data sources | Home + Data Loading |
| 3 | Performing descriptive analysis | Descriptive Analysis page |
| 4 | Conducting diagnostic analysis | Diagnostic Analysis page |
| 5 | Predictive analysis with ML models | Predictive Modeling page |
| 6 | Prescriptive analysis (optional) | Prescriptive Analysis page |
| – | Loading data | Data Loading page |
| – | Creating models / patterns | Predictive Modeling page |
| – | Applying models to new data | Apply Model page |
| – | Visualisations of analyses & predictions | Every page (Plotly) |

## ✨ Features

- **Data ingestion** — CSV, TSV, Excel, JSON, Parquet from upload or URL,
  plus seven curated sample datasets (Iris, Wine, Breast Cancer, Diabetes,
  California Housing, Titanic-style, Telco churn).
- **Descriptive analytics** — extended summary statistics (skew, kurtosis,
  variance, range), missing-value report, distribution & violin plots,
  categorical breakdowns and time-series resampling.
- **Diagnostic analytics** — Pearson/Spearman/Kendall correlation matrix,
  top driver ranking, IQR-based outlier scan, Welch t-test, one-way ANOVA,
  Chi-square independence test, scatter matrices.
- **Predictive analytics** — automatic task-type detection, leak-free
  preprocessing (imputation + scaling + one-hot encoding), benchmarking of
  Logistic / Linear / Ridge / Random Forest / Gradient Boosting / Decision
  Tree / KNN, cross-validation, confusion matrix, residual plot, feature
  importances.
- **Apply model** — manual single-row form, batch CSV scoring, downloadable
  predictions, and `.joblib` model export/import.
- **Prescriptive analytics** — automatic plain-language recommendations
  plus interactive one-feature sweeps and two-feature what-if heatmaps.
- **Pure-browser SaaS** — no server-side state, no databases; everything
  lives in `st.session_state`.

## 🚀 Quickstart

```bash
git clone <this repo>
cd IAD
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt                  # or: requirements.txt (runtime only)
streamlit run app.py
```

Streamlit will open the application at <http://localhost:8501>.

### Docker (production-like stack)

```bash
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000/docs |

See [docs/PHASE_10.md](docs/PHASE_10.md) for CI, pre-commit, and image publishing.

API (train, predict, upload): [PHASE_5](docs/PHASE_5.md). Advanced analytics, observability, exports: [PHASE_11](docs/PHASE_11.md) · [PHASE_12](docs/PHASE_12.md) · [PHASE_13](docs/PHASE_13.md).

### Quality gates (developers)

```bash
pip install -r requirements-dev.txt
pre-commit install
./scripts/ci.sh          # lint + typecheck + tests
```

## 🧭 Suggested walkthrough

1. **Home** — fill in the business case form (problem, objective, KPIs,
   stakeholders, data sources).
2. **📥 Data Loading** — pick the *Telco churn* sample (or upload your own
   file). Use the **Data hygiene** expander to drop duplicates or parse
   datetime columns if needed.
3. **📊 Descriptive Analysis** — explore distributions, missing-value heat
   bars and categorical break-downs.
4. **🔍 Diagnostic Analysis** — examine correlations, run a Welch t-test or
   Chi-square test on suspected drivers, and scan outliers.
5. **🤖 Predictive Modeling** — leave the suggested target (`churn`),
   keep all features, hit **Train models**. Inspect the leaderboard,
   feature importances and confusion matrix; download the model.
6. **🎯 Apply Model** — score a single hand-crafted record, then upload a
   batch CSV to download predictions.
7. **💡 Prescriptive Analysis** — read automatic recommendations, run a
   what-if sweep on `tenure_months` and a heatmap on
   `tenure_months × monthly_charges`.

## 🧱 Project layout

```
.
├── app.py                       # Main Streamlit entry / business case
├── pages/
│   ├── 1_📥_Data_Loading.py
│   ├── 2_📊_Descriptive_Analysis.py
│   ├── 3_🔍_Diagnostic_Analysis.py
│   ├── 4_🤖_Predictive_Modeling.py
│   ├── 5_🎯_Apply_Model.py
│   └── 6_💡_Prescriptive_Analysis.py
├── src/
│   ├── data_loader.py           # Sample datasets + file/URL loaders
│   ├── descriptive.py           # Stats, distributions, missing values
│   ├── diagnostic.py            # Correlations, hypothesis tests, outliers
│   ├── predictive.py            # Pipelines, training, evaluation, persistence
│   ├── prescriptive.py          # What-if scenarios + recommendations
│   └── utils.py                 # Session state, helpers
├── .streamlit/config.toml       # Theme & server settings
├── requirements.txt
└── README.md
```

## 🧠 How the workflow maps to data analytics theory

| Phase | Question answered | Techniques used in this app |
|---|---|---|
| Descriptive | *What happened?* | Summary stats, distributions, missing values |
| Diagnostic | *Why did it happen?* | Correlations, hypothesis tests, outliers |
| Predictive | *What will happen?* | Supervised ML (classification & regression) |
| Prescriptive | *What should we do?* | What-if scenarios, recommendations |

## 🛠️ Tech stack

- Streamlit `≥1.36` for the SaaS UI and multi-page navigation.
- pandas / numpy for data manipulation.
- scikit-learn for preprocessing, modelling, evaluation and cross-validation.
- scipy & statsmodels for hypothesis testing.
- Plotly Express for every visualisation (interactive zoom, hover, range).
- joblib for serialising trained pipelines.

## 🔮 Extending the app

- Plug in a new sample dataset → add an entry to
  `SAMPLE_DATASETS` in `src/data_loader.py`.
- Add another model → register it in `CLASSIFIERS` or `REGRESSORS` in
  `src/predictive.py`; the leaderboard picks it up automatically.
- Add a new analysis page → drop a `pages/N_<icon>_<Name>.py` file and
  Streamlit will detect it.

## 📄 License

For educational use as part of the "Data Analysis Tools" master's course.
# IAD

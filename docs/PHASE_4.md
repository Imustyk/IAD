# Phase 4 — Streamlit UX Refactor

## Architecture

Phase 4 introduces a **reusable frontend layer** under `iad/frontend/` without moving Streamlit’s page discovery path (`pages/` at repo root). This follows ADR-002 (strangler-fig): legacy `src/` analytics remain; UI is upgraded incrementally.

```
iad/frontend/
├── styles/          # Design tokens + CSS injection + dark mode
├── components/      # metric_cards, charts, tables, alerts, uploaders, model_cards, progress, navbar
├── layouts/         # setup_page(), dashboard home sections
└── services/        # session bridge, TrainingService ↔ legacy adapter
```

### Why `iad/frontend` and not `app/components`?

ADR-001: the Python package is `iad/`. Streamlit entry stays `app.py` to avoid shadowing the `app` module name.

### Design decisions

| Decision | Rationale |
|----------|-----------|
| CSS variables in `tokens.css` | Themeable light/dark without rebuilding Streamlit widgets |
| `setup_page()` bootstrap | One call replaces duplicated `set_page_config` + session + sidebar |
| `UnifiedTrainingReport` | Pages render one shape; engine toggles legacy vs enterprise |
| Enterprise ML default on | Phase 3 stack exposed via toggle; legacy path preserved |
| `@st.cache_data` on previews | Large datasets: cheap fingerprint cache for head previews |

## Directory tree (new / changed)

```
iad/frontend/
├── __init__.py
├── styles/
│   ├── tokens.css
│   ├── components.css
│   └── theme.py
├── components/
│   ├── metric_cards.py
│   ├── charts.py
│   ├── tables.py
│   ├── alerts.py
│   ├── uploaders.py
│   ├── model_cards.py
│   ├── progress.py
│   ├── navbar.py
│   └── layouts.py          # re-export
├── layouts/
│   ├── page.py
│   └── dashboard.py
└── services/
    ├── context.py
    └── training.py

app.py                        # KPI dashboard home
pages/*.py                    # all 6 pages use setup_page + components
tests/unit/frontend/          # component + bridge tests
```

## Integration

1. **`app.py`** — `page_config` → `render_home_dashboard()` with KPI cards and workflow grid.
2. **Every page** — calls `setup_page()` first (ensures `set_page_config` ordering).
3. **Predictive Modeling** — toggle `Enterprise ML engine`:
   - ON → `iad.ml.training.TrainingService` (XGBoost, auto-preprocessor, extended leaderboard)
   - OFF → `src.predictive.train_models` (original course pipeline)
4. **Session keys** — unchanged (`dataset`, `model_bundle`, `training_report`, …).

## Testing

```bash
pytest tests/unit/frontend/ -v
pytest --cov=iad --cov-report=term
streamlit run app.py
```

Manual checks:

- Dark mode toggle in sidebar persists across pages
- Data Loading → Descriptive → Predictive (both engines) → Apply Model
- Download `.joblib` still works

## Migration path

1. Phase 4 (this) — UI components + optional enterprise training on Predictive page
2. Phase 5 — wire SHAP/LIME tabs on Predictive page using `iad.ml.explainability`
3. Phase 5 — FastAPI backend sharing `TrainingService`

## Rollback

```bash
git checkout HEAD -- app.py pages/
rm -rf iad/frontend/styles iad/frontend/components iad/frontend/layouts iad/frontend/services
# Restore iad/frontend/__init__.py placeholder if needed
```

Set `IAD_UI_ENABLE_ENTERPRISE_ML=false` to default users to legacy training without code rollback.

## Production considerations

- Custom CSS uses `unsafe_allow_html=True` — keep HTML static; never interpolate user input into markdown HTML.
- Cache TTL: `IAD_UI_CACHE_TTL_SECONDS` (default 300).
- Streamlit reruns re-inject CSS; `page_config` must remain the first `st.*` call on each page.

# ML Copilot

ML Copilot is a web application that trains and evaluates machine
learning models on your tabular data from a plain-English objective.
Upload a CSV, tell it what you want ("build a model to predict customer
churn"), and it inspects the data, figures out the target column and
task type, trains and compares baseline models, tunes when it's worth
it, and reports back with real numbers — not a canned demo. Once a model
is saved, you can hand it new feature values and get a prediction back,
through the exact preprocessing it was trained with.

## What it does

- **Understands an objective.** You describe the goal in plain English;
  an LLM plans the workflow and asks for clarification if the target
  column is genuinely ambiguous, instead of guessing.
- **Inspects and profiles your data.** Shape, dtypes, missing values,
  duplicates, outliers, correlations, class balance — computed by
  pandas/scikit-learn, not invented by the model.
- **Detects the target and task type**, or asks you when it can't tell
  confidently which column you mean.
- **Trains and compares baseline models** (logistic regression / ridge,
  random forest, histogram gradient boosting) with cross-validation and
  a held-out test set, using a leakage-safe preprocessing pipeline.
- **Diagnoses over/underfitting** from the gap between training and
  validation performance, and explains what it found.
- **Tunes hyperparameters** with a bounded `RandomizedSearchCV` search —
  only when the diagnostics suggest it's worth the computation.
- **Saves the selected model** (preprocessing + estimator, as one
  artifact) and keeps a lightweight experiment log.
- **Explains the result**, grounded in the actual numbers the ML engine
  produced — the LLM narrates and interprets, it never invents a metric.
- **Predicts on new data.** Pick a saved model, enter feature values, and
  get a prediction back — class probabilities and a confidence score for
  classification, the predicted number for regression. The input form is
  generated from the saved model itself, so it works for any dataset
  without a line of dataset-specific code.

## Architecture

```
User
 ↓
Frontend (Next.js)
 ↓
FastAPI  ───────────────────────────  Projects / Datasets (CRUD, storage)
 ↓                                    (/copilot/* enters here)
LangGraph  ────────────────────────  orchestration: state, routing, limits
 ↓
LLM (OpenAI)  ─────────────────────  planning, tool selection, explanation
 ↓
MCP Client  ←──────────────────────  /models/* enters here instead
 ↓  (stdio, separate process)
MCP Server
 ↓
ML Engine (pandas / scikit-learn)  ── deterministic computation
```

A Copilot run goes through the LLM, because deciding *what to do next*
is a judgement call. Prediction isn't: the model, its features, and what
to run are all settled before the request arrives, so `/models/*` skips
the top half and calls the same MCP tools directly. Both paths end at
the same ML engine.

Each layer has one job, and the boundaries are enforced, not just
documented:

- **LLM** reasons, plans, chooses which tool to call next, and narrates
  the result. It never computes a metric, fits a model, or invents a
  number — every figure in the final response is copied directly from
  what the ML engine returned.
- **LangGraph** owns orchestration: a bounded graph (not an open-ended
  agent loop) with explicit stages, a human-in-the-loop pause/resume for
  target clarification, and hard iteration/tool-call/failure limits.
- **MCP** (Model Context Protocol) is the tool-calling boundary between
  the LLM/LangGraph process and the ML engine. Tools are discovered
  dynamically — nothing about which tools exist is hardcoded on the
  client side.
- **MCP server** runs as a *separate process*, in its own Python
  environment, and exposes the ML engine's capabilities as MCP tools.
  It's isolated because its dependencies (`mcp`'s server transport)
  genuinely conflict with the FastAPI app's pinned `starlette` version —
  not an arbitrary choice.
- **ML engine** is plain pandas/scikit-learn: dataset inspection, EDA,
  target detection, preprocessing, training, tuning, diagnostics, and
  persistence. Every function takes a DataFrame and returns structured
  data — no LLM, no MCP, no framework awareness.
- **FastAPI** is the HTTP boundary: project/dataset CRUD, file storage,
  a thin `/copilot/*` surface that starts/resumes/reads a LangGraph run,
  and `/models/*` for listing saved models, reading a model's expected
  feature schema, and predicting on new rows. It imports no pandas or
  scikit-learn and never opens a model artifact itself.

## Supported workflow

```
upload dataset
  → inspect (shape, dtypes, missing values, duplicates)
  → EDA (outliers, correlations, class balance)
  → target + task detection (or ask for clarification)
  → preprocessing (leakage-safe: fit only inside a CV fold / training split)
  → train & compare baseline models (cross-validated)
  → evaluate (task-appropriate metrics: F1/precision/recall/ROC-AUC or MAE/RMSE/R²)
  → diagnose over/underfitting
  → tune hyperparameters (only if the diagnostics justify it)
  → save the selected model
  → retrieve experiment results
  → explain the result
```

Then, any time afterwards:

```
select a saved model
  → read its feature schema (from the fitted pipeline inside the artifact)
  → enter feature values
  → predict — class probabilities + confidence, or a predicted number
```

## Project structure

```
backend/
    app/            FastAPI application — projects, datasets, /copilot/*, /models/*
    langgraph/       LangGraph orchestration graph, MCP client, LLM config
    mcp/             MCP server + ML engine (isolated Python environment)
    tests/           FastAPI app tests
frontend/
    app/             Next.js App Router entry point
    components/      UI, split by page/domain
    services/        typed API client (one file per backend resource)
    lib/             shared types and small utilities
```

## Running locally

### Backend

Requires two separate virtual environments — the FastAPI app and the MCP
server have genuinely conflicting dependencies (see
`backend/mcp/README.md`), so they're kept apart on purpose.

```bash
cd backend
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Windows
# .venv/bin/pip install -r requirements.txt         # macOS/Linux

python -m venv mcp/.venv
mcp/.venv/Scripts/pip install -r mcp/requirements.txt

cp .env.example .env    # fill in OPENAI_API_KEY

.venv/Scripts/python.exe -m alembic upgrade head   # creates app.db
.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

The API is now at `http://localhost:8000` (`/docs` for the OpenAPI UI).
The MCP server itself doesn't need to be started separately — the
LangGraph process spawns it as a subprocess per tool call.

### Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL, defaults to localhost:8000
npm run dev
```

Open `http://localhost:3000`.

### Environment variables

| File | Variable | Used by | Purpose |
|---|---|---|---|
| `backend/.env` | `OPENAI_API_KEY` | `backend/langgraph/llm/model.py` | Required. The LLM provider key. |
| `backend/.env` | `MLCOPILOT_OPENAI_MODEL` | same | Optional. Overrides the default model (`gpt-4o-mini`). |
| `backend/.env` | `MLCOPILOT_*` | `backend/app/core/config.py` | Optional. FastAPI app settings (`APP_NAME`, `DATABASE_URL`, `STORAGE_DIR`, `CORS_ORIGINS`) — see `.env.example` for the full list and defaults. |
| `frontend/.env.local` | `NEXT_PUBLIC_API_URL` | `frontend/services/api.ts` | The backend's base URL. Must be `NEXT_PUBLIC_`-prefixed to be readable by browser-side code. |

Both `backend/.env` and `frontend/.env.local` share one file per app on
purpose — see `backend/app/core/config.py`'s docstring for how the
backend settings class stays strict about typos in its own `MLCOPILOT_*`
namespace while tolerating the unrelated `OPENAI_API_KEY` living in the
same file.

## Testing

```bash
# FastAPI app
cd backend && .venv/Scripts/python.exe -m pytest tests/ -v

# LangGraph orchestration (mocked LLM + a few real MCP integration tests)
cd backend/langgraph && ../.venv/Scripts/python.exe -m pytest tests/ -v

# ML engine + MCP server
cd backend/mcp && .venv/Scripts/python.exe -m pytest tests/ -v

# Frontend
cd frontend && npm run lint && npx tsc --noEmit && npm run build
```

## Important architectural decisions

- **Two backend virtual environments, not one.** The MCP server's
  dependencies conflict with FastAPI's pinned `starlette`. Rather than
  force a shared environment (or drop one side's pin), the MCP server
  runs as an isolated subprocess with its own environment — this is also
  just how MCP's stdio transport is meant to be used.
- **Tools are discovered dynamically, never hardcoded.** LangGraph asks
  the MCP server what tools exist and offers all of them to the LLM at
  each stage; the graph's edges (not a tool-name lookup) decide what
  stage runs next.
- **The final result is grounded, not generated.** Every numeric/factual
  field in the Copilot's response is copied directly from what the ML
  engine returned; only the prose summary and caveats are LLM-authored,
  and even those are written from the real results, not from memory.
- **Leakage-safe preprocessing.** Preprocessing is built as an unfitted
  `ColumnTransformer` and only ever fit inside a `Pipeline`, on a
  training split or a single cross-validation fold — never on the whole
  dataset up front.
- **Prediction reuses the fitted pipeline; it never rebuilds one.** A
  saved artifact is the entire fitted `Pipeline`, and its preprocessing
  step carries values *learned* from the training split: each numeric
  column's imputation median, the scaler's mean and scale, and the exact
  category ordering the one-hot encoder assigned. Prediction reloads that
  object and calls `.predict()`. Refitting preprocessing on the incoming
  rows would change what each feature position means — the prediction
  wouldn't fail, it would just be quietly wrong.
- **The prediction form is generated from the model, not written in code.**
  Feature names and their training order, which columns are numeric vs.
  categorical, and each categorical column's valid values are all read off
  the fitted pipeline inside the artifact. No dataset's column names appear
  anywhere in the ML engine's callers, the API, or the UI.
- **Bounded orchestration, not an open-ended agent loop.** The
  orchestration graph is a DAG with one legitimate cycle (clarification,
  bounded by requiring a real human response each time), plus explicit
  iteration/tool-call/consecutive-failure limits as a safety net.
- **Human-in-the-loop clarification via LangGraph's `interrupt()`.** When
  the target column can't be confidently determined, the graph pauses
  (not restarts) and resumes from exactly that point once the frontend
  submits an answer.

## Evaluation and model-selection guarantees

- **One definition of F1:** classification uses macro-averaged precision,
  recall, and F1 in cross-validation, hyperparameter search, and held-out
  evaluation. Each class receives equal weight, so majority-class accuracy
  cannot hide poor minority-class performance. The API's `f1` field means
  macro F1 for both binary and multiclass targets.
- **Original labels survive training:** targets such as `yes`/`no` and
  nonconsecutive integer labels work without manual recoding. Binary ROC-AUC
  and PR-AUC use the probability column for `estimator.classes_[1]` as the
  positive class. Undefined AUC on a single-class holdout is omitted.
- **Tuning must earn promotion:** a completed search replaces the selected
  baseline only when its finite CV score strictly improves (higher F1,
  lower RMSE). Ties and worse scores retain the baseline run and artifact;
  the final report describes the model actually selected. Held-out test
  scores do not decide promotion. CV scores retain full precision for
  selection and are formatted only for display.

Regression tests cover imbalanced binary and multiclass scoring, string
labels through training/tuning/artifact reload, probability-column ordering,
non-finite tuning scores, and baseline retention in the final report.
Historical experiments created before this metric policy used different F1
averages and should be rerun before comparing their scores with new runs.

### Portfolio summary

Built a tabular ML copilot with FastAPI, LangGraph, and an isolated MCP
service: leakage-safe scikit-learn pipelines, reproducible cross-validation,
bounded hyperparameter search, persisted model inference, and deterministic
model promotion backed by regression tests. The LLM orchestrates the workflow
and explains measured results; Python owns evaluation and selection.

This is a development project, not a claim of production scale or benchmark
superiority. Live LLM behavior requires provider credentials; the integration
suite exercises real MCP and ML computation with a scripted LLM.

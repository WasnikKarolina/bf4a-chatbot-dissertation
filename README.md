# Bright Futures 4 All Website Chatbot

A hybrid chatbot built for the Bright Futures 4 All website.

The system supports:
- website navigation assistance
- FAQ answering using website content
- fixed intent-based responses
- hybrid retrieval over indexed website text
- KB fallback for curated topics
- AI-generated page summaries
- interactive BF4A quizzes
- page-opening actions for the website frontend

## What Is Included In The Submission
This repository already contains the built files needed to run the chatbot locally:
- trained intent model files in `backend/models/`
- retrieval index files in `backend/index/`
- chunked website data in `data/chunks/`
- page-level text used for summaries in `data/pages_text/`
- curated KB data in `data/kb/`

## Repository Structure
- `backend/` - FastAPI backend, chatbot services, models, index files and scripts
- `frontend/` - chatbot widget
- `data/` - page text, chunks, KB, training data, sitemap data
- `tests/` - evaluation code, datasets and generated results

## Prerequisites
- Python 3.12 recommended
- Windows PowerShell commands are used below

## Setup From Scratch
Follow these steps in order.

### Step 1: Open PowerShell in the repository root and create a virtual environment

```powershell
cd <path-to-project>\bf4a-chatbot-dissertation
```

```powershell
python -m venv .venv
```

### Step 2: Activate the virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Upgrade pip
```powershell
python -m pip install --upgrade pip
```

### Step 4: Install the required Python packages
```powershell
python -m pip install -r backend\requirements.txt
```

### Step 5: Install Playwright browser binaries
```powershell
python -m playwright install
```

### Step 6: Start the backend
In the current PowerShell window, start the backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload
```

Leave this terminal running.

### Step 7: Start the frontend
Open a second PowerShell window, go to the same repository root, and run:

```powershell
cd <path-to-project>\bf4a-chatbot-dissertation
```

```powershell
.\.venv\Scripts\python.exe -m http.server 8080 --directory frontend
```

Leave this terminal running as well.

### Step 8: Open the chatbot in a browser
Open:

```text
http://127.0.0.1:8080/
```

### Step 9: Check that all parts are running
You should now be able to:
- open the chatbot frontend at `http://127.0.0.1:8080/`
- access the backend health check at `http://127.0.0.1:8000/health`
- access the backend chat API: `http://127.0.0.1:8000/chat`
- send messages through the chatbot UI


## What To Expect On First Run
- The chatbot should start without retraining or rebuilding indexes.
- The first summary request may be slower because a local summarisation model may download or load.
- Some retrieval features may also trigger the first local download/load of Hugging Face model files if they are not already cached.

## Optional: Run The Automated Evaluation And Generate The Report
If you want to run the backend evaluation used for the report, install `pytest` first:

```powershell
python -m pip install pytest
```

Then run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests\evaluation\test_backend_performance.py
```
Then run:

```powershell
.\.venv\Scripts\python.exe tests\evaluation\backend_evaluation.py
```

This writes:
- `tests/results/report.json`

## Optional: Retrain And Rebuild Data
You do not need to run these steps for local demonstration.

Only rerun them if you intentionally change the website data, training data, or index files.

Run: 
```powershell
.\.venv\Scripts\python.exe backend\scripts\train_intent_model.py
```

Then run these in order:

```powershell
.\.venv\Scripts\python.exe backend\scripts\get_sitemap_urls.py
.\.venv\Scripts\python.exe backend\scripts\crawl_site.py
.\.venv\Scripts\python.exe backend\scripts\extract_text.py
.\.venv\Scripts\python.exe backend\scripts\chunk_text.py
.\.venv\Scripts\python.exe backend\scripts\build_tfidf_index.py
.\.venv\Scripts\python.exe backend\scripts\build_embeddings_index.py
```

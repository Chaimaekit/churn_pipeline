# Moroccan Telecom Churn Prediction Pipeline

> **Enterprise-grade ML pipeline for predicting customer churn in Moroccan telecom markets, enriched with real-time social media sentiment analysis from Facebook (Darija/Arabic support).**

---

## Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Facebook Scraping](#2-facebook-scraping)
  - [3. Feedback Analysis](#3-feedback-analysis)
  - [4. Model Training](#4-model-training)
  - [5. Real-time Inference](#5-real-time-inference)
- [API Endpoints](#-api-endpoints)
- [Model Performance](#-model-performance)
- [Strengths](#-strengths)
- [Troubleshooting](#-troubleshooting)

---

## Overview

This pipeline addresses a critical business problem in the Moroccan telecom sector: **predicting which customers are likely to churn** before they actually leave. What makes this solution unique is its integration of:

- **Traditional telecom metrics** (call duration, charges, service calls)
- **Social media sentiment signals** scraped from Orange Maroc's Facebook page
- **LLM-powered Darija (Moroccan Arabic) text analysis** for complaint categorization
- **Multi-model ensemble** (Logistic Regression, XGBoost, LightGBM, CatBoost)

### Business Impact

| Metric | Value |
|--------|-------|
| **Target Market** | Moroccan telecom subscribers |
| **Languages Supported** | English, French, Darija (Moroccan Arabic) |
| **Social Data Source** | Facebook (Orange Maroc official page) |
| **Models Trained** | 4 (LR, XGBoost, LightGBM, CatBoost) |
| **Champion Model** | CatBoost |
| **Decision Threshold** | 0.40 (optimized for recall) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION LAYER                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────────┐  │
│  │  PostgreSQL  │  │  CSV/Excel   │  │  Facebook Scraper (Selenium)     │  │
│  │   Database   │  │   Uploads    │  │  • Posts & Reels                 │  │
│  └──────┬───────┘  └──────┬───────┘  │  • Comments extraction           │  │
│         └─────────────────┘            │  • Darija text support           │  │
│                                        └────────────────┬─────────────────┘  │
└─────────────────────────────────────────────────────────┼────────────────────┘
                                                          │
┌─────────────────────────────────────────────────────────▼────────────────────┐
│                      FEATURE ENGINEERING LAYER                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │  data_loader.py   │  │  processing.py    │  │  feedback_analyse.py    │  │
│  │  • Load telecom   │  │  • Clean & encode │  │  • Groq LLM API         │  │
│  │    subscriber     │  │  • Feature eng.   │  │  • Darija sentiment     │  │
│  │    data           │  │  • Handle missing │  │  • Complaint intensity  │  │
│  └──────────────────┘  └──────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                                          │
┌─────────────────────────────────────────────────────────▼────────────────────┐
│                      MODEL TRAINING LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────────┐  │
│  │   Logistic  │  │   XGBoost   │  │  LightGBM   │  │    CatBoost      │  │
│  │ Regression  │  │  (tuned)    │  │   (tuned)   │  │   (Champion)     │  │
│  │  Baseline   │  │             │  │             │  │                  │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────────────┘  │
│                                                                             │
│  • Stratified K-Fold CV (k=3)                                               │
│  • RandomizedSearchCV for hyperparameter tuning                           │
│  • scale_pos_weight for class imbalance                                     │
│  • SHAP explainability integration                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                                          │
┌─────────────────────────────────────────────────────────▼────────────────────┐
│                      EVALUATION & SERVING LAYER                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────────┐  │
│  │   evaluate.py     │  │    train.py       │  │      main.py (FastAPI)  │  │
│  │  • ROC/PR curves  │  │  • Model persis- │  │  • /health              │  │
│  │  • Benchmark      │  │    tence         │  │  • /run_pipeline        │  │
│  │    comparison     │  │  • Serialization │  │  • /prediction/realtime │  │
│  │  • SHAP global    │  │                  │  │  • /prediction/{id}     │  │
│  │    importance     │  │                  │  │  • /reputation          │  │
│  └──────────────────┘  └──────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
churn_pipeline/
│
├── main.py                      # FastAPI application + orchestration
├── data_loader.py               # Data ingestion (CSV/Excel/PostgreSQL/Facebook JSON)
├── processing.py                # Feature engineering & preprocessing pipeline
├── train.py                     # Model training with hyperparameter tuning
├── evaluate.py                  # Performance benchmarking & SHAP explainability
├── feedback_analyse.py          # LLM-based Darija sentiment analysis (Groq API)
├── scrape_facebook.py           # Selenium-based Facebook comment scraper
├── requirements.txt             # Python dependencies
│
├── data/                        # Data directory (gitignored)
│   ├── comments_*.json             # Scraped Facebook comments
│   ├── processed_feedback.csv      # LLM-analyzed feedback
│   ├── train_with_feedback.csv     # Training dataset
│   └── test_with_feedback.csv      # Test dataset
│
├── models/                      # Serialized models
│   └── champion_catboost.pkl       # Production champion model
│
└── output/                      # Generated reports & visualizations
    ├── model_performance_report.md
    ├── model_metrics_comparison.png
    ├── model_comparison_curves.png
    └── champion_feature_importance.png
```

---

## Features

### Core ML Pipeline
- **4 Model Ensemble**: Logistic Regression (baseline), XGBoost, LightGBM, CatBoost (champion)
- **Hyperparameter Tuning**: RandomizedSearchCV with Stratified K-Fold
- **Class Imbalance Handling**: `scale_pos_weight` computed from training distribution
- **SHAP Explainability**: Global feature importance for business stakeholders
- **Threshold Optimization**: 0.40 threshold tuned for operational recall

### Social Media Integration
- **Facebook Scraper**: Selenium-based automation for posts & reels
- **Comment Extraction**: Handles both post comments and reel comment drawers
- **Darija Support**: Native Moroccan Arabic text extraction
- **Anti-Detection**: undetected-chromedriver + stealth configurations

### LLM Analysis
- **Groq API Integration**: `openai/gpt-oss-120b` for Darija understanding
- **Sentiment Classification**: Positive / Neutral / Negative
- **Complaint Categorization**: pricing, service_quality, customer_support, churn_intent
- **Intensity Scoring**: 1-5 scale for complaint severity

### Production API
- **FastAPI**: Async REST endpoints
- **Real-time Inference**: Single customer prediction
- **Batch Lookup**: Customer ID-based retrieval from CSV
- **Brand Reputation**: Aggregated social sentiment KPIs
- **Health Checks**: Service monitoring endpoint

---

## Installation

### Prerequisites
- Python 3.10+
- Chrome browser (for Facebook scraping)
- Groq API key (for LLM analysis)

### Step 1: Clone & Setup

```bash
git clone <repository-url>
cd churn_pipeline
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Environment Variables

Create a `.env` file in the project root:

```env
# Groq API (for Darija sentiment analysis)
GROQ_API_KEY=your_groq_api_key_here

# Facebook Credentials (for scraping)
FB_EMAIL=your_facebook_email
FB_PASSWORD=your_facebook_password

# PostgreSQL (optional - for database ingestion)
POSTGRES_HOST=localhost
POSTGRES_DB=telecom_db
POSTGRES_USER=admin
POSTGRES_PASSWORD=secret
POSTGRES_PORT=5432
```

### Step 4: Chrome Setup (Windows)

Download Chrome portable and update the path in `scrape_facebook.py`:
```python
CHROME_BINARY = r"C:\Users\<user>\Downloads\chrome-win\chrome.exe"
```

---

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PAGE_NAME` | `orangname` | Facebook page to scrape |
| `MAX_POSTS` | `50` | Number of posts/reels to collect |
| `MAX_RETRIES` | `2` | Comment button click retries |
| `target_threshold` | `0.40` | Churn probability cutoff |
| `CHROME_BINARY` | — | Path to Chrome executable |

### Blocked Usernames (UI Filtering)

The scraper automatically filters Facebook UI elements masquerading as usernames:
```python
BLOCKED_USERNAMES = {
    "Follow", "Like", "Reply", "Share", "Comment", "Orange", "Meta AI",
    "Groups", "Find friends", "Home", "Create", "Menu", "Notifications",
    # ... (see full list in scrape_facebook.py)
}
```

---

## Usage

### 1. Data Ingestion

#### From CSV/Excel
```python
from data_loader import load_real_telecom_data

df = load_real_telecom_data("data/subscribers.csv")
# Supports .csv, .xlsx, .xls
```

#### From PostgreSQL
```python
from data_loader import load_from_postgres

query = "SELECT * FROM subscribers WHERE active = 1"
df = load_from_postgres(query)
```

#### From Facebook Scraping
```bash
python scrape_facebook.py
# Manual login required in browser window
# Outputs: data/comments_YYYYMMDD_HHMMSS.json
```

### 2. Facebook Scraping

The scraper handles both **posts** and **reels**:

| Content Type | Behavior |
|-------------|----------|
| **Reels** | Clicks comment button → opens comment drawer → extracts |
| **Posts** | Comments visible by default → clicks "View more comments" → extracts |

**Output format:**
```json
[
  {
    "customer_id": "e819425d08fb",
    "post_url": "https://facebook.com/reel/...",
    "username": " افنان",
    "text": "واش كاين لعرض 6*20",
    "scraped_at": "2026-06-20T13:07:23.251884"
  }
]
```

### 3. Feedback Analysis

Process scraped comments through Groq LLM:

```bash
python feedback_analyse.py
```

**Categories detected:**
- `pricing` — price complaints, billing issues
- `service_quality` — network, coverage, speed
- `customer_support` — support experience complaints
- `churn_intent` — explicit switching/cancellation intent

**Output:** `data/processed_feedback.csv`

### 4. Model Training

Run the full pipeline:

```bash
python main.py
```

Or via API:
```bash
curl -X POST "http://localhost:8000/run_pipeline"   -H "Content-Type: application/json"   -d '{
    "train_path": "data/train_with_feedback.csv",
    "test_path": "data/test_with_feedback.csv",
    "threshold": 0.40
  }'
```

**Pipeline stages:**
1. Load & clean data
2. Engineer features (total minutes, charges, usage intensity, etc.)
3. Encode categoricals
4. Train 4 models with cross-validation
5. Benchmark & select champion
6. Generate SHAP explainability plots
7. Serialize champion model

### 5. Real-time Inference

#### Single Customer Prediction
```bash
curl -X POST "http://localhost:8000/prediction/realtime"   -H "Content-Type: application/json"   -d '{
    "customer_id": "CUST_001",
    "state": "CA",
    "account_length": 120,
    "area_code": 510,
    "international_plan": "No",
    "voice_mail_plan": "Yes",
    "number_vmail_messages": 15,
    "total_day_minutes": 200.5,
    "total_day_calls": 105,
    "total_day_charge": 34.09,
    "total_eve_minutes": 180.2,
    "total_eve_calls": 95,
    "total_eve_charge": 15.32,
    "total_night_minutes": 220.1,
    "total_night_calls": 110,
    "total_night_charge": 9.90,
    "total_intl_minutes": 12.5,
    "total_intl_calls": 3,
    "total_intl_charge": 3.38,
    "customer_service_calls": 2,
    "feedback_text": "الخدمة غالية بزاف",
    "complaint_intensity": 4
  }'
```

**Response:**
```json
{
  "customer_id": "CUST_001",
  "churn_probability": 0.72,
  "churn_risk_level": "High",
  "action_required": true
}
```

#### Batch Customer Lookup
```bash
curl "http://localhost:8000/prediction/CUST_002667"
```

#### Brand Reputation
```bash
curl "http://localhost:8000/reputation"
```

**Response:**
```json
{
  "total_social_records_evaluated": 150,
  "net_reputation_score": -15.3,
  "brand_health_status": "At Risk / Negative Operational Friction",
  "average_complaint_intensity": 3.2,
  "sentiment_distribution": {
    "positive": 20,
    "neutral": 45,
    "negative": 85
  },
  "top_operational_complaints": {
    "pricing": 60,
    "service_quality": 45,
    "churn_intent": 30,
    "customer_support": 15
  }
}
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/run_pipeline` | POST | Execute full training pipeline |
| `/prediction/realtime` | POST | Predict churn for single customer |
| `/prediction/{customer_id}` | GET | Lookup & predict from batch file |
| `/reputation` | GET | Brand reputation KPIs from social data |

---

## Model Performance

### Champion: CatBoost

| Metric | Score |
|--------|-------|
| **ROC-AUC** | ~0.92 |
| **PR-AUC** | ~0.88 |
| **Accuracy** | ~0.89 |
| **Precision (Churn)** | ~0.85 |
| **Recall (Churn)** | ~0.82 |
| **F1-Score (Churn)** | ~0.83 |

### Benchmark Comparison

Generated automatically in `output/`:
- `model_comparison_curves.png` — ROC & PR curves for all models
- `model_metrics_comparison.png` — Bar chart comparison
- `model_performance_report.md` — Markdown report
- `champion_feature_importance.png` — SHAP global importance

---


## Strengths

1. **Novel Social Integration**: First known pipeline combining telecom CDR data with Moroccan Darija Facebook sentiment
2. **Production-Ready API**: FastAPI with async support, health checks, and proper error handling
3. **Explainability**: SHAP integration provides business-friendly feature importance
4. **Class Imbalance Handling**: `scale_pos_weight` dynamically computed from training data
5. **Multi-Model Benchmarking**: Systematic comparison prevents model bias
6. **Darija NLP**: LLM-powered analysis of Moroccan Arabic dialect (rare in existing tools)


## Troubleshooting

### Facebook Scraping Issues

| Symptom | Solution |
|---------|----------|
| "Comment button not found" | Increase `time.sleep()` after page load; check if reel loaded |
| "No comments extracted" | Verify login cookies; check if post is public |
| Multiple Chrome processes | Set `use_subprocess=False` in undetected-chromedriver |
| StaleElementReference | Add try/except with retry logic; re-query DOM |

### Model Training Issues

| Symptom | Solution |
|---------|----------|
| `KeyError: churn` | Ensure target column is named `churn` or `churn_flag` |
| `ValueError: could not convert` | Check for non-numeric values in numeric columns |
| CUDA out of memory | Reduce `n_estimators` or use CPU-only XGBoost |

### API Issues

| Symptom | Solution |
|---------|----------|
| "Model is not loaded" | Run `/run_pipeline` first to train & serialize |
| "Customer ID not found" | Verify ID exists in `data/test_with_feedback.csv` |
| CORS errors | Add `CORSMiddleware` to FastAPI app |

---


## Acknowledgments

- [Groq](https://groq.com/) for LLM API access
- [SHAP](https://shap.readthedocs.io/) for model explainability
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [CatBoost](https://catboost.ai/) for the champion gradient boosting implementation

---

> **Built for the Moroccan telecom market. For questions open an issue or reach out to me.**

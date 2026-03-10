# 🔍 Intelligent Supplier Risk Intelligence System

> An NLP + RAG-powered supply chain risk monitoring system that ingests supplier PDF documents, extracts geopolitical, financial, and compliance risk signals using transformer-based embeddings and named entity recognition, and provides a conversational Q&A interface over indexed documents.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.1-green)
![FAISS](https://img.shields.io/badge/FAISS-CPU-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

Supply chain disruptions cost businesses trillions annually. This system automates supplier risk monitoring by:

- **Ingesting** supplier PDF documents (contracts, audit reports, financial filings)
- **Extracting** geopolitical, financial, compliance, and operational risk signals using NLP
- **Scoring** each supplier on a 0–100 risk scale across four risk dimensions
- **Indexing** all content into a FAISS vector store using transformer embeddings
- **Answering** contextual questions via a RAG pipeline (LangChain + OpenAI or local fallback)
- **Visualizing** risk dashboards with interactive Plotly charts in Streamlit

---

## 🏗️ Architecture

```
Supplier PDFs
     │
     ▼
┌─────────────────┐
│  PDF Ingestion  │  PyMuPDF → text extraction → chunking
│  (ingestion.py) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────────┐
│  Risk Extractor │     │   Vector Store        │
│ (risk_extractor)│     │  sentence-transformers│
│ NER + Keywords  │     │  → FAISS index        │
│ Geo/Fin/Comp/Op │     │  (rag_pipeline.py)    │
└────────┬────────┘     └──────────┬───────────┘
         │                         │
         ▼                         ▼
┌─────────────────────────────────────────────┐
│            Streamlit Dashboard              │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Risk Scores  │  │  RAG Q&A Assistant   │ │
│  │ + Charts     │  │  LangChain + OpenAI  │ │
│  └──────────────┘  └──────────────────────┘ │
└─────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/supplier-risk-intelligence.git
cd supplier-risk-intelligence
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Set up OpenAI API key
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```
> Without an API key, the system runs in **local mode** — documents are still indexed and retrieved, but answers use extracted snippets instead of GPT-generated responses.

### 5. Generate sample supplier PDFs (for demo)
```bash
python scripts/generate_sample_pdfs.py
```

### 6. Launch the Streamlit app
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
supplier-risk-intelligence/
├── app.py                        # Streamlit dashboard (main entry point)
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment variable template
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── ingestion.py              # PDF text extraction and chunking
│   ├── risk_extractor.py         # NLP risk signal extraction + scoring
│   └── rag_pipeline.py           # FAISS vector store + LangChain RAG Q&A
│
├── data/
│   └── sample_pdfs/              # Place supplier PDFs here
│
└── scripts/
    └── generate_sample_pdfs.py   # Demo PDF generator
```

---

## 🧠 Key Components

### 1. PDF Ingestion (`src/ingestion.py`)
- Extracts text from supplier PDFs using **PyMuPDF**
- Splits text into overlapping chunks (500 words, 50-word overlap)
- Preserves metadata: supplier name, page number, source file

### 2. Risk Signal Extraction (`src/risk_extractor.py`)
- Keyword-based NLP detection across **4 risk dimensions**:
  | Dimension | Examples |
  |-----------|---------|
  | 🌍 Geopolitical | sanctions, export controls, trade war, OFAC |
  | 💰 Financial | bankruptcy, credit downgrade, liquidity, debt |
  | ⚖️ Compliance | violations, fraud, bribery, audit findings |
  | ⚙️ Operational | supply disruption, sole supplier, factory fire |
- Severity scoring: Low / Medium / High per signal
- Weighted aggregation to an **Overall Risk Score (0–100)**

### 3. RAG Pipeline (`src/rag_pipeline.py`)
- Embeds all chunks using **`all-MiniLM-L6-v2`** (sentence-transformers, runs locally)
- Indexes embeddings in a **FAISS** flat L2 index
- Retrieves top-k relevant chunks per query
- Generates answers via **LangChain + GPT-3.5** (or local fallback)

### 4. Streamlit Dashboard (`app.py`)
- **Risk Dashboard**: Score cards, bar charts, pie charts, radar plots
- **AI Q&A**: Conversational interface with source attribution
- **Signal Details**: Drill-down into extracted risk signals per supplier

---

## 📊 Risk Scoring Methodology

```
Overall Score = (Geopolitical × 0.30) + (Financial × 0.30)
              + (Compliance × 0.25) + (Operational × 0.15)

Score 0–29  → 🟢 Low Risk
Score 30–59 → 🟡 Medium Risk
Score 60–100 → 🔴 High Risk
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| PDF Parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector Search | FAISS (CPU) |
| RAG Framework | LangChain |
| LLM (optional) | OpenAI GPT-3.5-turbo |
| NLP/NER | Keyword NLP (extensible to spaCy) |
| Dashboard | Streamlit |
| Charting | Plotly |
| Data Processing | Pandas, NumPy |

---

## 🔧 Extending the System

**Add new risk categories:**
Edit `GEOPOLITICAL_KEYWORDS` / `FINANCIAL_KEYWORDS` etc. in `src/risk_extractor.py`

**Use spaCy NER:**
Install `spacy` and `en_core_web_sm`, then add entity-based extraction on top of keyword matching

**Add more data sources:**
Extend `src/ingestion.py` to handle news RSS feeds, SEC EDGAR filings, or web scraping

**Swap the LLM:**
Replace `ChatOpenAI` in `rag_pipeline.py` with any LangChain-compatible model (Anthropic Claude, local Ollama, etc.)

---

## 📸 Screenshots

| Risk Dashboard | Q&A Assistant | Signal Details |
|---|---|---|
| Risk scores, radar chart, bar chart | Conversational RAG interface | Keyword-highlighted signals |

---

## 📄 License

MIT License — feel free to use and extend this project.

---

## 🙋 About

Built to demonstrate applied NLP and RAG techniques in a real-world supply chain risk context.
Combines document understanding, semantic search, and generative AI for intelligent risk monitoring.

# AI-Powered Plagiarism Detection System

An end-to-end AI-powered plagiarism detection system built using **FastAPI**, **Streamlit**, and a **Retrieval-Augmented Generation (RAG)** pipeline. The system combines semantic search, hybrid retrieval, and Large Language Model (LLM) verification to detect **Exact Copy**, **Near Copy**, and **Paraphrased** content while generating detailed plagiarism reports.

---

## Features

- AI-powered plagiarism detection using semantic similarity
- Hybrid retrieval with **FAISS** and **BM25**
- Cross-Encoder reranking for improved retrieval accuracy
- LLM-based verification using **Ollama** or **OpenAI**
- Supports **PDF**, **DOCX**, and **TXT** documents
- Interactive Streamlit dashboard
- Reference library (corpus) management
- Scan history with detailed analytics
- Downloadable PDF plagiarism reports
- Configurable embedding models and LLM providers

---

## Architecture

```
Document Upload
        │
        ▼
Text Extraction
        │
        ▼
Text Preprocessing
        │
        ▼
Semantic Chunking
        │
        ▼
Sentence Embeddings
        │
        ▼
Hybrid Retrieval
 (FAISS + BM25)
        │
        ▼
Cross-Encoder Reranking
        │
        ▼
LLM Verification
        │
        ▼
Classification
(Exact Copy / Near Copy /
 Paraphrased / Original)
        │
        ▼
PDF Report & Dashboard
```

---

## Tech Stack

### Backend

- FastAPI
- SQLAlchemy
- SQLite
- Sentence Transformers
- Transformers
- FAISS
- BM25
- NLTK
- FPDF2

### Frontend

- Streamlit
- Plotly

### AI Models

- Sentence Transformers
- Cross-Encoder
- Ollama
- OpenAI GPT Models

---

## Project Structure

```
AI_Plagiarism_Detector/
│
├── backend/
│   ├── api/
│   ├── database/
│   ├── llm/
│   ├── rag/
│   ├── schemas/
│   ├── services/
│   ├── utils/
│   ├── vectorstore/
│   ├── app.py
│   └── config.py
│
├── frontend/
│   ├── pages/
│   ├── api_client.py
│   └── Home.py
│
├── storage/
│   ├── corpus/
│   ├── uploads/
│   ├── reports/
│   └── faiss_index/
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository.

```bash
git clone https://github.com/Aditya-Sonwane/AI-Plagiarism-Detection-System.git

cd AI_Plagiarism_Detector
```

Install dependencies.

```bash
pip install -r requirements.txt
```

Create the environment file.

```bash
cp .env.example .env
```

Update your API keys and configuration inside `.env`.

---

## Running the Application

### Start the Backend

```bash
uvicorn backend.app:app --reload
```

Backend API:

```
http://localhost:8000
```

API Documentation:

```
http://localhost:8000/docs
```

---

### Start the Frontend

```bash
streamlit run frontend/Home.py
```

Application:

```
http://localhost:8501
```

---

## Usage

1. Upload reference documents to build the plagiarism library.
2. Upload a document for plagiarism detection.
3. The system extracts and preprocesses the document.
4. Relevant content is retrieved using FAISS and BM25.
5. Retrieved candidates are reranked using a Cross-Encoder.
6. An LLM verifies semantic similarity and classifies matches.
7. View detailed results and download the generated PDF report.

---

## Configuration

The application can be configured through the `.env` file.

Example:

```env
PLAG_LLM_PROVIDER=openai
PLAG_OPENAI_API_KEY=your_api_key
PLAG_OPENAI_MODEL=gpt-4o-mini

PLAG_API_BASE_URL=http://localhost:8000

PLAG_EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
PLAG_EMBEDDING_DIM=384
```

You can also switch to a local Ollama model by changing the provider.

---

## Supported File Formats

- PDF
- DOCX
- TXT

---

## Output

The system provides:

- Overall plagiarism percentage
- Classification breakdown
- Sentence-level plagiarism analysis
- Similarity score
- Confidence score
- Source document information
- Downloadable PDF report
- Scan history

---

## Future Improvements

- OCR support for scanned PDF documents
- Internet source plagiarism detection
- Multi-language plagiarism detection
- Highlight plagiarized text directly inside PDF reports
- User authentication and role management
- Cloud deployment with Docker and Kubernetes

---


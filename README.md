# 📄 RAG Document QA System

A Retrieval-Augmented Generation (RAG) based system for querying and reasoning over multiple documents using Large Language Models (LLMs), embeddings, and vector search.

---

## 🚀 Features

* 📂 Upload and process multiple documents (PDFs)
* 🔍 Semantic search using embeddings
* 🧠 Retrieval-Augmented Generation (RAG) pipeline
* ⚡ FastAPI backend for scalable APIs
* 🌐 Frontend interface for interaction
* 📊 Supports multi-document querying and comparison

---

## 🏗️ Architecture

1. **Document Ingestion**

   * Upload PDFs
   * Extract text and split into chunks

2. **Embedding Generation**

   * Convert text chunks into vector embeddings

3. **Vector Storage**

   * Store embeddings in a vector database (ChromaDB)

4. **Retrieval**

   * Retrieve relevant chunks based on user query

5. **Generation**

   * Pass retrieved context + query to LLM for answer generation

---

## 🛠️ Tech Stack

* **Backend**: FastAPI, Python
* **Frontend**: (React / Vite or similar)
* **LLM & AI**: OpenAI / Transformers
* **Vector DB**: ChromaDB
* **Libraries**: LangChain, Sentence Transformers, PyPDF

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/rag-document-qa-system.git
cd rag-document-qa-system
```

### 2. Backend setup

```bash
cd backend
pip install -r requirements.txt
```

### 3. Run backend

```bash
uvicorn main:app --reload
```

Backend will run on:

```
http://127.0.0.1:8000
```

---

## 🌐 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 🔑 Environment Variables

Create a `.env` file in `backend` folder:

```
OPENAI_API_KEY=your_api_key_here
```

---

## 📌 API Docs

Once backend is running, open:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Example Workflow

1. Upload a document
2. Ask a question
3. System retrieves relevant chunks
4. LLM generates contextual answer

---

## 📈 Future Improvements

* Add authentication
* Improve chunking strategy
* Add streaming responses
* Deploy on cloud (AWS / GCP)

---

## 🙌 Author

**Srikar Reddy**

---

## ⭐ Note

This project demonstrates practical implementation of RAG pipelines using modern LLM tools and frameworks.

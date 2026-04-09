# 🚀 RAG Document QA System

A production-style **Retrieval-Augmented Generation (RAG)** system that enables intelligent querying and reasoning over multiple documents using LLMs, embeddings, and vector search.

---

## ✨ Overview

This project demonstrates how to build an end-to-end RAG pipeline that:

* Ingests documents (PDFs)
* Converts them into vector embeddings
* Stores them in a vector database
* Retrieves relevant context based on queries
* Generates accurate answers using an LLM

---

## 🧠 Key Highlights

* 🔍 Semantic search over documents using embeddings
* 📄 Multi-document support with contextual understanding
* ⚡ FastAPI backend for scalable API handling
* 🧩 Modular architecture (routes, services, pipelines)
* 🧠 LLM-powered responses with contextual grounding
* 🔄 Extendable for real-world production use cases

---

## 🏗️ System Architecture

```
User Query
     ↓
Embedding Model
     ↓
Vector Database (ChromaDB)
     ↓
Relevant Context Retrieval
     ↓
LLM (Generation)
     ↓
Final Answer
```

---

## 🛠️ Tech Stack

| Category         | Tools / Libraries                |
| ---------------- | -------------------------------- |
| Backend          | FastAPI, Python                  |
| AI/ML            | LangChain, Sentence Transformers |
| LLM              | OpenAI / Transformers            |
| Vector Store     | ChromaDB                         |
| Document Parsing | PyPDF                            |
| Frontend         | React (Vite)                     |

---

## ⚙️ Setup Instructions

### 🔹 Clone Repository

```bash
git clone https://github.com/srikar-reddyy/rag-document-qa-system.git
cd rag-document-qa-system
```

---

### 🔹 Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

> ⚠️ If installation fails, manually install missing packages:

```bash
pip install pypdf PyMuPDF
```

---

### 🔹 Run Backend Server

```bash
uvicorn main:app --reload
```

Access API:
👉 http://127.0.0.1:8000/docs

---

### 🔹 Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

---

## 🔐 Environment Variables

Create a `.env` file inside `backend`:

```
OPENAI_API_KEY=your_api_key_here
```

---

## 🧪 How It Works

1. Upload documents (PDFs)
2. System extracts and chunks text
3. Converts chunks into embeddings
4. Stores vectors in ChromaDB
5. User submits a query
6. Relevant chunks are retrieved
7. LLM generates context-aware answer

---

## 📌 Example Use Cases

* Document-based Q&A systems
* Research assistants
* Knowledge base search
* Enterprise document intelligence

---

## 📈 Future Improvements

* Add authentication & user sessions
* Streaming responses (real-time answers)
* UI improvements and chat interface
* Cloud deployment (AWS / GCP / Docker)

---

## 👨‍💻 Author

**Srikar Reddy**

---

## 🌟 Why This Project Matters

This project showcases practical understanding of:

* Retrieval-Augmented Generation (RAG)
* LLM integration
* Vector databases
* Scalable backend development

---

## ⭐ If you find this useful

Give it a ⭐ on GitHub!

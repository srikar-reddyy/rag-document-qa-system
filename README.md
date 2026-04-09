# 📚 RAG-Multidocument

A powerful, full-stack application for engaging with and analyzing multiple documents using Retrieval-Augmented Generation (RAG). 
Upload files (PDFs, Text, etc.), ask insightful questions across your entire document base, and compare specific chunks and references directly!

## ✨ Features

- **Multi-Document Chat:** Ask questions and get answers synthesized from all your uploaded documents.
- **Intelligent Chunking:** Advanced text splitting algorithms to retain semantic context.
- **Compare Pipeline:** Analyze and compare different documents side-by-side in real time. 
- **100% Local Inference Support:** Integrated with **Ollama** for completely private, offline, and secure LLM generation.
- **Scalable Vector Database:** Uses **ChromaDB** for lightning-fast embeddings & retrieval.
- **Interactive Interface:** Clean React UI with resizable panels, PDF/Text viewers, and integrated document highlights.

## 🛠️ Tech Stack

**Frontend:**
- React.js + Tailwind CSS
- Resizable Panels & Custom PDF Viewers

**Backend:**
- FastAPI (Python)
- Ollama (LLM Integration)
- ChromaDB (Vector Database)
- Advanced text parsing, chunking, and search pipelines

## 🚀 Getting Started

### Prerequisites
- **Python 3.9+**
- **Node.js (v16+) & npm**
- **[Ollama](https://ollama.ai/)** (downloaded and running locally)

### 1️⃣ Backend Setup
Open a terminal and navigate to the backend specific directory.
`ash
cd backend

# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # (Windows)
# source venv/bin/activate # (Mac/Linux)

# Install requirements
pip install -r requirements.txt

# Start the API server
python -m uvicorn main:app --reload
`
*API runs at http://localhost:8000*

### 2️⃣ Frontend Setup
Open a separate terminal and navigate to the frontend directory.
`ash
cd frontend

# Install Node modules
npm install

# Start the UI
npm start
`
*App UI runs at http://localhost:3000*

## 📂 Project Structure Overview

- ackend/rag/ - Core retrieval logic for embedding, chunking, and vectors.
- ackend/routes/ - FastAPI endpoints for chat, comparison, file upload, and debugging.
- rontend/src/components/ - Interactive chat components, file uploaders, document viewer panes.
- rontend/src/pages/ - Parent application routing views.

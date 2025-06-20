# PDF Chat App with LLM (Ollama + LangChain + Flask)

A simple web app to chat with the contents of any PDF file using an LLM. Built using:

- **Flask** – lightweight backend server
- **LangChain** – document parsing, embedding, and retrieval
- **FAISS** – vector store for fast similarity search
- **Ollama** – runs LLMs locally (e.g., LLaMA3)
- **HuggingFace Transformers** – embeddings for PDF chunks

---

# Features

- ✅ Upload a PDF file
- ✅ Ask questions about its content
- ✅ Ask general knowledge questions (not limited to the PDF)
- ✅ Local and private (runs entirely on your machine)
- ✅ Uses Ollama to avoid API costs or rate limits
- ✅ Shows "Thinking..." indicator while answering

---


## 🚀 Getting Started

### 1. Clone the repo

### bash
git clone https://github.com/AliC137/AI_assistant.git
cd flask-app.py



# Project Structure

AIAvicenna/
│
├── flask-app.py                  # Main Flask app
├── uploads/                # Uploaded PDFs (not tracked by Git)
├── faiss_indexes/

# Make sure Ollama is installed

ollama run llama3

# You can use any LLM that supports Ollama

ollama pull llama3
ollama pull mistral
ollama pull gemma


# Just change the model name in flask-app.py

llm = OllamaLLM(model="mistral")


# Pip Requirements

flask==2.3.2
langchain-community==0.0.100
langchain-ollama==0.0.8
faiss-cpu==1.7.4
sentence-transformers==2.2.2
huggingface-hub==0.15.1
werkzeug==2.3.4



from flask import Flask, request, render_template_string, jsonify
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from werkzeug.utils import secure_filename
import os, hashlib

app = Flask(__name__)
UPLOAD_FOLDER = "./uploads"
INDEX_FOLDER = "./faiss_indexes"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(INDEX_FOLDER, exist_ok=True)

# Initialize embedding and LLM
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
llm = OllamaLLM(model="llama3")

qa_chain = None

# HTML with chat and AJAX file upload
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>AI Assistant Avicenna Chat</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background-color: #f8f9fa; }
    #chat-box {
      background-color: white;
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 15px;
      height: 400px;
      overflow-y: auto;
      margin-bottom: 15px;
    }
    .message { margin-bottom: 10px; }
    .user { text-align: right; }
    .user .bubble {
      display: inline-block;
      background: #d1e7dd;
      color: #0f5132;
      padding: 10px 15px;
      border-radius: 15px 15px 0 15px;
    }
    .bot .bubble {
      display: inline-block;
      background: #e2e3e5;
      color: #41464b;
      padding: 10px 15px;
      border-radius: 15px 15px 15px 0;
    }
  </style>
</head>
<body class="container py-4">

  <h2 class="mb-4">AIAvicenna: Chat Assistant</h2>

  <div class="mb-3">
    <label for="file" class="form-label">Upload a PDF</label>
    <form id="uploadForm" enctype="multipart/form-data" class="input-group">
      <input class="form-control" type="file" name="file" id="file" accept=".pdf,.docx" required>
      <button class="btn btn-primary" type="submit">Upload</button>
    </form>
    <div id="upload-status" class="form-text mt-1 text-success"></div>
  </div>

  <div id="chat-box"></div>

  <div class="input-group">
    <input id="user-input" class="form-control" placeholder="Ask a question..." />
    <button class="btn btn-success" onclick="sendMessage()" id="send-btn">Send</button>
  </div>
  <div id="loading" class="form-text mt-2 text-muted" style="display:none;">Thinking...</div>

  <script>
    document.getElementById("uploadForm").addEventListener("submit", async function(e) {
  e.preventDefault();

  const fileInput = document.getElementById("file");
  const status = document.getElementById("upload-status");
  const file = fileInput.files[0];

  if (!file) {
    status.innerText = "Please select a file.";
    status.className = "form-text text-danger";
    return;
  }

  const allowedTypes = [".pdf", ".docx"];
  const ext = file.name.split('.').pop().toLowerCase();
  if (!allowedTypes.includes("." + ext)) {
    status.innerText = "Unsupported file type. Please upload a .pdf or .docx file.";
    status.className = "form-text text-danger";
    return;
  }

  status.innerText = "Uploading...";
  status.className = "form-text text-muted";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData
    });
    const result = await response.json();

    if (response.ok) {
      status.innerText = result.message;
      status.className = "form-text text-success";
    } else {
      status.innerText = result.message || "Upload failed.";
      status.className = "form-text text-danger";
    }
  } catch (err) {
    status.innerText = "An unexpected error occurred.";
    status.className = "form-text text-danger";
  }
});
  </script>
</body>
</html>
"""


def file_hash(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/upload", methods=["POST"])
def upload():
    global qa_chain

    file = request.files.get("file")
    if not file:
        return jsonify({"message": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()

    # Only accept PDF and DOCX
    if ext not in [".pdf", ".docx"]:
        return jsonify({"message": "Unsupported file type. Please upload a .pdf or .docx file."}), 400

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    index_name = file_hash(file_path)
    index_dir = os.path.join(INDEX_FOLDER, index_name)
    os.makedirs(index_dir, exist_ok=True)

    try:
        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        else:
            # Load based on file type
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(file_path)

            documents = loader.load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(index_dir)

        retriever = db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        return jsonify({"message": f"{ext.upper()} file uploaded and processed successfully."})
    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500

@app.route("/chat", methods=["POST"])
def chat():
    global qa_chain
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"answer": "Please enter a question."})

    try:
        if qa_chain:
            answer = qa_chain.run(question)
            # Fallback to general answer if vague or "I don't know"
            if not answer.strip() or "i don't know" in answer.lower() or "not mentioned" in answer.lower():
                answer = llm.invoke(question)
        else:
            answer = llm.invoke(question)

        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)

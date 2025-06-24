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
<html lang="en" data-bs-theme="light">
<head>
  <meta charset="utf-8">
  <title>AI Assistant Avicenna Chat</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  body {
    background-color: var(--bs-body-bg);
    color: var(--bs-body-color);
  }

  #chat-box {
    background-color: var(--bs-body-bg);
    border: 1px solid var(--bs-border-color);
    border-radius: 12px;
    padding: 15px;
    height: 400px;
    overflow-y: auto;
    margin-bottom: 15px;
  }

  .message {
    margin-bottom: 15px;
  }

  .bubble {
    padding: 10px 15px;
    border-radius: 15px;
    max-width: 80%;
  }

  .user.bubble {
    background: #d1e7dd;
    color: #0f5132;
  }

  .bot.bubble {
    background: #e2e3e5;
    color: #41464b;
  }

  .avatar {
    width: 36px;
    height: 36px;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background-color: #dee2e6;
  }

  /* New styling for file input, buttons and user input */
  #uploadForm input[type="file"],
  #uploadForm button,
  #user-input,
  #send-btn {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-color: #444444 !important;
  }

  /* Placeholder color for user input */
  #user-input::placeholder {
    color: #bbbbbb !important;
  }

  /* Focus outlines */
  #uploadForm input[type="file"]:focus,
  #user-input:focus {
    outline: 2px solid #0d6efd;
    background-color: #121212 !important;
    color: #fff !important;
  }

  @media (prefers-color-scheme: dark) {
    html {
      color-scheme: dark;
    }

    body {
      background-color: #121212;
      color: #f1f1f1;
    }

    #chat-box {
      background-color: #1e1e1e;
      border-color: #333;
    }

    .user.bubble {
      background-color: #265d4a;
      color: #d1ffd5;
    }

    .bot.bubble {
      background-color: #3a3a3a;
      color: #e0e0e0;
    }

    .avatar {
      background-color: #3a3a3a;
    }
  }
</style>
</head>
<body class="container py-4">

  <h2 class="mb-4">AIAvicenna: Chat Assistant</h2>

  <div class="mb-3">
    <label for="file" class="form-label">Upload a PDF or DOCX</label>
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

    document.getElementById("user-input").addEventListener("keydown", function(event) {
      if (event.key === "Enter") {
        event.preventDefault();
        sendMessage();
      }
    });

    async function sendMessage() {
      const input = document.getElementById("user-input");
      const chatBox = document.getElementById("chat-box");
      const loading = document.getElementById("loading");
      const question = input.value.trim();
      if (!question) return;

      chatBox.innerHTML += `
        <div class="message d-flex justify-content-end align-items-start">
          <div class="user bubble me-2">${question}</div>
          <div class="avatar">🧑</div>
        </div>
      `;
      input.value = "";
      loading.style.display = "block";

      try {
        const response = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: question })
        });
        const data = await response.json();
        if (data.answer) {
          chatBox.innerHTML += `
            <div class="message d-flex align-items-start">
              <div class="avatar me-2">🤖</div>
              <div class="bot bubble">${marked.parse(data.answer)}</div>
            </div>
          `;
        }
      } catch (error) {
        chatBox.innerHTML += `
          <div class="message d-flex align-items-start">
            <div class="avatar me-2">🤖</div>
            <div class="bot bubble">An error occurred while processing your request.</div>
          </div>
        `;
      }

      loading.style.display = "none";
      chatBox.scrollTop = chatBox.scrollHeight;
    }
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
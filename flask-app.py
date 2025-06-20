from flask import Flask, request, render_template_string, jsonify
from langchain_community.document_loaders import PyPDFLoader
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
<title>Upload PDF and Chat</title>
<h1>Upload PDF</h1>
<form id="uploadForm" enctype="multipart/form-data">
  <input type="file" name="file" required>
  <input type="submit" value="Upload">
</form>
<p id="upload-status"></p>

<hr>
<h2>Chat</h2>
<div id="chat-box" style="border:1px solid #ccc; padding:10px; height:300px; overflow-y:scroll;"></div>
<input id="user-input" placeholder="Ask a question..." style="width:80%;">
<button onclick="sendMessage()">Send</button>
<p id="loading" style="display:none;">Thinking...</p>

<script>
document.getElementById("uploadForm").addEventListener("submit", async function(e) {
    e.preventDefault();
    const formData = new FormData(this);
    const status = document.getElementById("upload-status");
    status.innerText = "Uploading...";
    
    const response = await fetch("/upload", {
        method: "POST",
        body: formData
    });

    const result = await response.json();
    status.innerText = result.message;
});

async function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const loading = document.getElementById("loading");

    if (!input.value) return;

    chatBox.innerHTML += "<p><b>You:</b> " + input.value + "</p>";
    loading.style.display = "block";

    const res = await fetch("/chat", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({question: input.value})
    });
    const data = await res.json();
    chatBox.innerHTML += "<p><b>Bot:</b> " + data.answer + "</p>";
    loading.style.display = "none";
    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;
}
</script>
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
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    index_name = file_hash(file_path)
    index_dir = os.path.join(INDEX_FOLDER, index_name)
    os.makedirs(index_dir, exist_ok=True)

    try:
        if os.path.exists(os.path.join(index_dir, "index.faiss")):
            db = FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
        else:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(index_dir)

        retriever = db.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        return jsonify({"message": "PDF uploaded and processed successfully."})
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

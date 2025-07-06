import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import faiss
from langchain_groq import ChatGroq
import tempfile
import numpy as np
import shutil

# Load env vars
load_dotenv()

app = Flask(__name__)
CORS(app)

processed_documents_data = {}
conversation_history_data = {}
llm_instance = None
embedding_model = None

# --- Init models only once ---
def init_models():
    global llm_instance, embedding_model
    if llm_instance is None:
        llm_instance = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192",
            temperature=0.7
        )
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

init_models()

# --- PDF utils ---
def get_pdf_text(pdf_paths):
    text = ""
    for path in pdf_paths:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + "\n"
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=800, chunk_overlap=100, length_function=len
    )
    return splitter.split_text(text)

def build_faiss_index(chunks):
    embeddings = embedding_model.embed_documents(chunks)
    vectors = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, chunks

def get_relevant_docs(question, index, chunks, k=4):
    query = embedding_model.embed_query(question)
    D, I = index.search(np.array([query]).astype("float32"), k)
    return [chunks[i] for i in I[0]]

# --- API ---
@app.route("/process_pdfs", methods=["POST"])
def process_pdfs():
    user_id = "default_user"
    if 'pdfs' not in request.files:
        return jsonify({"error": "No PDF files uploaded."}), 400

    files = request.files.getlist('pdfs')
    if not files:
        return jsonify({"error": "No files selected."}), 400

    temp_paths = []
    try:
        for file in files:
            tmp_dir = tempfile.mkdtemp()
            path = os.path.join(tmp_dir, file.filename)
            file.save(path)
            temp_paths.append((tmp_dir, path))

        pdf_paths = [p for _, p in temp_paths]
        raw_text = get_pdf_text(pdf_paths)
        if not raw_text.strip():
            return jsonify({"error": "No readable text found in PDFs."}), 400

        chunks = get_text_chunks(raw_text)
        chunks = chunks[:300]  # Limit for memory
        index, stored_chunks = build_faiss_index(chunks)

        processed_documents_data[user_id] = {
            'faiss_index': index,
            'text_chunks': stored_chunks
        }
        conversation_history_data[user_id] = []
        return jsonify({"message": "PDFs processed!", "user_id": user_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        for dir_path, _ in temp_paths:
            shutil.rmtree(dir_path, ignore_errors=True)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    question = data.get("question")
    style = data.get("response_style", "Detailed")
    user_id = data.get("user_id", "default_user")

    if not question:
        return jsonify({"error": "No question provided."}), 400
    if user_id not in processed_documents_data:
        return jsonify({"error": "Please process a PDF first."}), 400

    try:
        faiss_index = processed_documents_data[user_id]['faiss_index']
        text_chunks = processed_documents_data[user_id]['text_chunks']
        history = conversation_history_data.get(user_id, [])

        # Rephrase question if history exists
        standalone_q = question
        if history:
            hist = "\n".join([f"{m['type']}: {m['content']}" for m in history[-6:]])
            prompt = f"""Given the conversation, rephrase the follow-up:
Chat:
{hist}

Q: {question}
Rephrased Q:"""
            try:
                resp = llm_instance.invoke(prompt)
                if resp and resp.content:
                    standalone_q = resp.content.strip()
            except: pass

        docs = get_relevant_docs(standalone_q, faiss_index, text_chunks)
        context = "\n\n".join(docs)

        answer_prompt = f"""
You are an intelligent assistant helping a user understand their documents.
Only use the context below. Respond in a {style.lower()} way.
Say "Information not available" if unsure.

Context:
{context}

Question:
{question}

Answer:
"""
        response = llm_instance.invoke(answer_prompt)
        final_answer = response.content.strip()

        history.append({"type": "human", "content": question})
        history.append({"type": "ai", "content": final_answer})
        conversation_history_data[user_id] = history[-10:]  # trim history

        return jsonify({"answer": final_answer, "chat_history": history}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/clear_cache", methods=["POST"])
def clear():
    user_id = "default_user"
    processed_documents_data.pop(user_id, None)
    conversation_history_data.pop(user_id, None)
    return jsonify({"message": "Session cleared."}), 200

if __name__ == '__main__':
    app.run(debug=False, host="0.0.0.0", port=5000)

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter # Still useful for chunking
from langchain.embeddings import HuggingFaceEmbeddings # Using directly for embeddings
import faiss # Direct FAISS import
from langchain_groq import ChatGroq # Direct LLM interaction
import tempfile
import numpy as np # For numerical operations with embeddings
import json # For serializing chat history

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# Global store for processed data and LLM instances (simplified for single user)
# In a real app, manage this per-user/session
processed_documents_data = {} # Stores {'text_chunks': [], 'embeddings': np.array}
conversation_history_data = {} # Stores chat history as a list of dicts per user
llm_instance = None # Will be initialized once
embedding_model = None # Will be initialized once

# Initialize LLM and Embedding Model once
def init_models():
    global llm_instance, embedding_model
    if llm_instance is None:
        llm_instance = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model_name="llama3-70b-8192"
        )
    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Call model initialization when the app starts
with app.app_context():
    init_models()

# 1. Extract text from PDFs
def get_pdf_text(pdf_paths):
    text = ""
    for pdf_path in pdf_paths:
        try:
            with pdfplumber.open(pdf_path) as pdf_file:
                for page in pdf_file.pages:
                    content = page.extract_text()
                    if content:
                        text += content + "\n" # Add newline for better separation
        except Exception as e:
            app.logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise ValueError(f"Failed to extract text from {os.path.basename(pdf_path)}. Is it a valid PDF?")
    return text

# 2. Split text into chunks
def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return splitter.split_text(text)

# 3. Build vectorstore and store chunks
def build_faiss_index(text_chunks):
    # Generate embeddings for all chunks
    chunk_embeddings = embedding_model.embed_documents(text_chunks)
    chunk_embeddings_np = np.array(chunk_embeddings).astype('float32')

    # Create FAISS index
    dimension = chunk_embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension) # L2 distance (Euclidean) for similarity
    index.add(chunk_embeddings_np)

    return index, text_chunks

# Function to retrieve relevant documents using FAISS
def get_relevant_documents(query_text, faiss_index, text_chunks, k=4):
    query_embedding = embedding_model.embed_query(query_text)
    query_embedding_np = np.array(query_embedding).astype('float32').reshape(1, -1)

    distances, indices = faiss_index.search(query_embedding_np, k) # Search for top k similar chunks
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks

# --- API Endpoints ---

@app.route('/process_pdfs', methods=['POST'])
def process_pdfs():
    if 'pdfs' not in request.files:
        return jsonify({"error": "No PDF files uploaded"}), 400

    pdf_files = request.files.getlist('pdfs')
    if not pdf_files:
        return jsonify({"error": "No PDF files selected"}), 400

    pdf_paths = []
    temp_dirs = [] # To keep track of temporary directories for cleanup
    user_id = "default_user" # Simplified user ID

    try:
        for pdf_file in pdf_files:
            # Save the uploaded PDF to a temporary file
            temp_dir = tempfile.mkdtemp()
            temp_pdf_path = os.path.join(temp_dir, pdf_file.filename)
            pdf_file.save(temp_pdf_path)
            pdf_paths.append(temp_pdf_path)
            temp_dirs.append(temp_dir) # Store the temporary directory path

        raw_text = get_pdf_text(pdf_paths)
        if not raw_text.strip():
            return jsonify({"error": "Could not extract any meaningful text from PDFs. They might be scanned images or empty."}), 400

        text_chunks = get_text_chunks(raw_text)
        faiss_index, indexed_text_chunks = build_faiss_index(text_chunks)

        # Store processed data globally for this user_id
        processed_documents_data[user_id] = {
            'faiss_index': faiss_index,
            'text_chunks': indexed_text_chunks
        }
        conversation_history_data[user_id] = [] # Initialize empty chat history

        return jsonify({"message": "PDFs processed successfully!", "user_id": user_id}), 200

    except Exception as e:
        app.logger.error(f"Error in process_pdfs: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up temporary files and directories
        for temp_dir_path in temp_dirs:
            if os.path.exists(temp_dir_path):
                # tempfile.mkdtemp creates a directory, so use rmtree
                import shutil
                shutil.rmtree(temp_dir_path)


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_question = data.get('question')
    response_style = data.get('response_style', 'Detailed')
    user_id = data.get('user_id', 'default_user') # Retrieve user ID

    if not user_question:
        return jsonify({"error": "No question provided"}), 400

    if user_id not in processed_documents_data:
        return jsonify({"error": "No PDFs processed for this session. Please upload PDFs first."}), 400

    faiss_index = processed_documents_data[user_id]['faiss_index']
    text_chunks = processed_documents_data[user_id]['text_chunks']
    chat_history = conversation_history_data.get(user_id, [])

    try:
        # Step 1: Rephrase standalone question (if there's chat history)
        # This part still conceptually uses an LLMChain pattern, but we're doing it manually
        standalone_question = user_question
        if chat_history:
            history_str = "\n".join([f"{msg['type']}: {msg['content']}" for msg in chat_history])
            condense_prompt = (
                f"Given the following conversation and a follow-up question, rephrase the follow-up to be a standalone question.\n\n"
                f"Chat History:\n{history_str}\n\n"
                f"Follow-Up Question:\n{user_question}\n\n"
                f"Rephrased Standalone Question:"
            )
            # Directly call the LLM
            try:
                condensed_response = llm_instance.invoke(condense_prompt)
                standalone_question = condensed_response.content.strip()
                if not standalone_question or "follow-up question" in standalone_question.lower(): # Basic check for bad rephrasing
                    standalone_question = user_question
            except Exception as e:
                app.logger.warning(f"Could not rephrase question with LLM, using original: {e}")
                standalone_question = user_question # Fallback

        # Step 2: Retrieve relevant documents using the standalone question
        relevant_docs = get_relevant_documents(standalone_question, faiss_index, text_chunks)
        context = "\n\n".join(relevant_docs)

        # Step 3: Formulate the final prompt for the answer
        doc_prompt_template = """
You are an intelligent assistant helping a user understand their documents.

Your job is to:
- Carefully read the context extracted from the uploaded PDFs.
- Provide maximum accuracy.
- Only use the context. Do not guess.
- If the answer isn't in the documents, say: "The information is not available in the provided documents."

Respond to the following prompt in a {response_style} manner:

Context:
{context}

Question:
{question}

Answer:
"""
        final_prompt = doc_prompt_template.format(
            response_style=response_style.lower(), # Ensure style is lowercase for prompt
            context=context,
            question=user_question # Use original question for final answer, but context from standalone
        )

        # Step 4: Call the LLM to get the answer
        ai_response_obj = llm_instance.invoke(final_prompt)
        ai_answer = ai_response_obj.content.strip()

        # Step 5: Update chat history
        chat_history.append({"type": "human", "content": user_question})
        chat_history.append({"type": "ai", "content": ai_answer})
        conversation_history_data[user_id] = chat_history

        # Return the answer and history (serialized)
        return jsonify({"answer": ai_answer, "chat_history": chat_history}), 200

    except Exception as e:
        app.logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route('/clear_cache', methods=['POST'])
def clear_backend_cache():
    user_id = "default_user" # Again, use a proper session ID in a real app
    if user_id in processed_documents_data:
        del processed_documents_data[user_id]
    if user_id in conversation_history_data:
        del conversation_history_data[user_id]
    return jsonify({"message": "Backend cache and memory cleared."}), 200


if __name__ == '__main__':
    # Initialize models on startup
    app.run(debug=True, port=5000) # Run Flask app on port 5000
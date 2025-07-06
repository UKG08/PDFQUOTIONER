import os
import uuid
import PyPDF2
import textwrap
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder='templates')
CORS(app)

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in a .env file.")
client = Groq(api_key=groq_api_key)

user_data = {}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_pdfs', methods=['POST'])
def process_pdfs():
    if 'pdfs' not in request.files:
        return jsonify({"error": "No PDF files provided"}), 400

    files = request.files.getlist('pdfs')
    if not files:
        return jsonify({"error": "No PDF files selected"}), 400

    user_id = str(uuid.uuid4())
    user_data[user_id] = {'pdf_chunks': [], 'chat_history': []}
    all_text = ""
    processed_file_names = []

    try:
        for file in files:
            if file.filename == '':
                continue
            if file and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                processed_file_names.append(filename)
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    all_text += page.extract_text() + "\n"

        if not all_text.strip():
            return jsonify({"error": "No readable text found in the uploaded PDFs."}), 400

        chunks = textwrap.wrap(all_text, CHUNK_SIZE, break_long_words=False, replace_whitespace=False)
        final_chunks = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                overlap = chunks[i-1][-CHUNK_OVERLAP:]
                final_chunks.append(overlap + chunk)
            else:
                final_chunks.append(chunk)

        user_data[user_id]['pdf_chunks'] = final_chunks

        return jsonify({
            "message": f"Successfully processed {len(processed_file_names)} PDF(s) and created {len(final_chunks)} chunks. You can now ask questions!",
            "user_id": user_id,
            "processed_files": processed_file_names,
            "chunks_count": len(final_chunks)
        }), 200

    except PyPDF2.errors.PdfReadError:
        return jsonify({"error": "Invalid PDF file. Could not read."}), 400
    except Exception as e:
        app.logger.error(f"Error processing PDFs: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during PDF processing: {str(e)}"}), 500

@app.route('/chat', methods=['POST'])
def chat_question():
    data = request.get_json()
    question = data.get('question')
    response_style = data.get('response_style', 'detailed')
    user_id = data.get('user_id')

    if not user_id or user_id not in user_data:
        return jsonify({"error": "Session not found. Please upload PDFs first."}), 400
    if not question:
        return jsonify({"error": "No question provided."}), 400

    pdf_chunks = user_data[user_id]['pdf_chunks']
    chat_history = user_data[user_id]['chat_history']

    if not pdf_chunks:
        return jsonify({"error": "No PDFs processed for this session. Please upload PDFs."}), 400

    try:
        MAX_CONTEXT_CHARS = 7000

        relevant_context = ""
        for chunk in pdf_chunks:
            if len(relevant_context) + len(chunk) < MAX_CONTEXT_CHARS:
                relevant_context += chunk + "\n\n"
            else:
                break

        if not relevant_context:
            relevant_context = "No sufficient context could be retrieved from the documents for this question."

        base_system_prompt = (
            "You are PDFER, an intelligent AI assistant designed to help users understand their PDF documents. "
            "Your primary goal is to provide accurate, comprehensive, and helpful answers **strictly based on the information provided in the document context below.**\n\n"
            "**Key Rules for Answering:**\n"
            "1.  **Strictly Document-Based:** If the answer is not explicitly stated or inferable from the provided document content, you MUST state: 'I cannot find this information in the uploaded documents.' Do NOT use external knowledge or invent facts.\n"
            "2.  **Clarity & Coherence:** Ensure your answers are clear, well-structured, and easy to understand.\n"
            "3.  **Completeness:** Provide a complete answer to the user's question, addressing all parts of the query.\n"
            "4.  **Formatting:** Use Markdown for better readability (e.g., bolding, bullet points, numbered lists, code blocks) where appropriate.\n"
            "5.  **Avoid Redundancy:** Do not repeat information unnecessarily."
        )

        style_instruction = ""
        if response_style == "detailed":
            style_instruction = (
                "**Response Style: Detailed & Comprehensive**\n"
                "Provide a thorough and in-depth explanation. Elaborate on concepts, include relevant specifics, and ensure a complete overview of the topic as found in the document. Break down complex information into digestible parts."
            )
        elif response_style == "concise":
            style_instruction = (
                "**Response Style: Concise & To-the-Point**\n"
                "Formulate a brief and succinct answer. Get straight to the essential information without unnecessary elaboration or fluff. Summarize main facts efficiently."
            )
        elif response_style == "key points":
            style_instruction = (
                "**Response Style: Key Points & Summarized**\n"
                "Extract and present the core information as a bulleted list of key points. Each bullet point should capture a distinct and important piece of information related to the question. Do not write full paragraphs; use short, impactful phrases or sentences."
            )

        final_system_message = f"{base_system_prompt}\n\n{style_instruction}"

        messages = [
            {"role": "system", "content": final_system_message},
            {"role": "user", "content": f"Here is the document context you must use to answer my question:\n\n---\n{relevant_context}\n---\n\nMy question is: {question}\n\nBased ONLY on the provided document context, please provide the best possible answer according to the specified style."}
        ]

        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=1000
        )

        ai_answer = chat_completion.choices[0].message.content

        user_data[user_id]['chat_history'].append({"role": "user", "content": question})
        user_data[user_id]['chat_history'].append({"role": "assistant", "content": ai_answer})

        return jsonify({"answer": ai_answer}), 200

    except Exception as e:
        app.logger.error(f"Error during chat completion: {e}", exc_info=True)
        return jsonify({"error": f"An unexpected error occurred during chat: {str(e)}"}), 500

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    user_id = request.get_json().get('user_id')
    if user_id in user_data:
        del user_data[user_id]
        return jsonify({"message": "PDF memory cleared and session reset."}), 200
    return jsonify({"message": "No active session to clear or session already cleared."}), 200

# Remove the __main__ block if you are using Gunicorn (recommended for Render)
# if __name__ == '__main__':
#     # For local development:
#     # Flask's built-in server is NOT for production
#     port = int(os.environ.get("PORT", 5000))
#     app.run(debug=True, host='0.0.0.0', port=port)

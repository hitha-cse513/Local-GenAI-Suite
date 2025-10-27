from flask import Flask, render_template, request, jsonify
import requests
import json
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama2"

# ============== CHATBOT ===================
@app.route('/chatbot')
def chatbot_page():
    return render_template('chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat_with_bot():
    user_message = request.json['message']
    payload = {"model": MODEL_NAME, "prompt": user_message, "stream": False}
    r = requests.post(OLLAMA_API_URL, json=payload)
    text = ""
    for line in r.iter_lines():
        part = json.loads(line.decode())
        text += part.get("response", "")
    return jsonify({"response": text})

# ============== PDF Q&A ===================
vectordb = None
qa_chain = None

@app.route('/pdf-qa')
def pdf_qa_page():
    return render_template('pdf_qa.html')

@app.route('/api/load_pdf', methods=['POST'])
def load_pdf():
    global vectordb, qa_chain
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        file.save(tmp.name)
        try:
            print("PDF file saved at", tmp.name)
            loader = PyPDFLoader(tmp.name)
            pages = loader.load_and_split()
            print(f"Loaded {len(pages)} pages")
            if not pages:
                return jsonify({"message": "❌ No text pages found in PDF."})
            embeddings = OllamaEmbeddings(model="llama2")
            print("Embeddings created")
            vectordb = FAISS.from_documents(pages, embeddings)
            print("FAISS vector DB initialized")
            llm = OllamaLLM(model="llama2")
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever())
            print("QA chain built")
            return jsonify({"message": f"✅ PDF loaded with {len(pages)} pages."})
        except Exception as e:
            print("Exception:", str(e))
            return jsonify({"message": f"❌ Error: {str(e)}"})
        
@app.route('/api/ask_pdf', methods=['POST'])
def ask_pdf():
    global qa_chain
    question = request.json['question']
    if qa_chain is None:
        return jsonify({"answer": "⚠️ Please upload a PDF first."})
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"] if isinstance(result, dict) and "result" in result else str(result)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"❌ Error: {str(e)}"})


# ============== CODE GENERATION ==============
@app.route('/codegen')
def codegen_page():
    return render_template('codegen.html')

@app.route('/api/generate_code', methods=['POST'])
def generate_code():
    prompt = request.json['prompt']
    gen_pipe = pipeline("text-generation", model="GuillenLuis03/PyCodeGPT")
    gen_output = gen_pipe(prompt, max_length=64, temperature=0.7, num_return_sequences=1)[0]['generated_text']
    return jsonify({"code": gen_output})

# ============== IMAGE GENERATION ==============
@app.route('/imagegen')
def imagegen_page():
    return render_template('imagegen.html')

@app.route('/api/generate_image', methods=['POST'])
def generate_image():
    prompt = request.json['prompt']
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    pipe = pipe.to('cuda')
    img = pipe(prompt).images[0]
    img_path = tempfile.mktemp(suffix=".png")
    img.save(img_path)
    return jsonify({"image_path": img_path})

# ============== HOME PAGE ===================
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)


# 🧠 Gen AI Suite: All-in-One Local AI Workspace

## Overview

The **Gen AI Suite** is a Flask-based, multi-tool web application that unifies cutting-edge AI models for **conversation, document analysis, code generation, and image creation** — all running locally on your system.

It integrates the power of **Ollama LLaMA 2**, **LangChain**, **Stable Diffusion**, and **Hugging Face Transformers** into one intuitive interface, perfect for researchers, developers, and AI enthusiasts.

🌐 **Local App URL:**
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Source

* **LLaMA 2 (Ollama)** — for natural language conversations and reasoning.
* **LangChain + FAISS** — for PDF question answering with vector-based retrieval.
* **Hugging Face Transformers** — for on-demand Python code generation.
* **Stable Diffusion 2.1** — for AI image synthesis powered by diffusion models.
* **Flask** — provides a clean and dynamic web interface.

---

## Features

* 💬 **LLaMA 2 Chatbot** — Chat naturally with a local AI assistant.
* 📄 **PDF Q&A Assistant** — Upload a PDF and ask context-aware questions.
* 💻 **Python Code Generator** — Generate ready-to-use Python scripts from natural language prompts.
* 🖼️ **Image Generator** — Create AI-generated artwork and visuals using Stable Diffusion.
* 🎨 **Unified Modern Interface** — Beautiful, glassmorphic design with gradient blue aesthetics.
* ⚙️ **Runs Entirely Offline** — All models run locally; no API keys or cloud usage required.

---

## Key Modules

### 1. 🗨️ LLaMA 2 Chatbot

* Built with **Ollama LLaMA 2** for local inference.
* Engages in conversational dialogue and reasoning.
* Simple, responsive chat UI with message bubbles.

### 2. 📚 PDF Q&A Assistant

* Uses **LangChain’s RetrievalQA** pipeline.
* Extracts and embeds PDF text using **OllamaEmbeddings** + **FAISS**.
* Accurately retrieves relevant paragraphs for user queries.

### 3. 💻 Python Code Generator

* Powered by **Hugging Face Transformers** and the model `GuillenLuis03/PyCodeGPT`.
* Converts plain English prompts into executable Python code.

### 4. 🖼️ Image Generator

* Uses **Stable Diffusion 2.1** via Hugging Face **Diffusers**.
* Generates beautiful, high-resolution images from text prompts.
* Saves images to temporary local files and displays them instantly.

---

## Installation & Usage

### 🔧 Clone the Repository

```bash
git clone https://github.com/hitha-cse513/Local-GenAi-Suite.git
cd  Local-GenAi-Suite
```

---

### 📦 Install Dependencies

Ensure **Python 3.9+** is installed, then run:

```bash
pip install -r requirements.txt
```

---

### 🧠 Set Up Ollama (for LLaMA 2)

Install Ollama (for macOS, Windows, or Linux) from [https://ollama.ai](https://ollama.ai).

Then, pull the **LLaMA 2 model**:

```bash
ollama pull llama2
ollama serve
```

Keep this service running in the background.

```bash
ollama run llama2
```
---

### ▶️ Run the App

```bash
python app.py
```

Once the server starts, open your browser and go to:

```bash
http://127.0.0.1:5000
```

Explore the following tools:

| Tool                | Description                          | URL         |
| ------------------- | ------------------------------------ | ----------- |
| 💬 Chatbot          | Conversational AI powered by LLaMA 2 | `/chatbot`  |
| 📄 PDF Q&A          | Document-based question answering    | `/pdf_qa`   |
| 💻 Code Generator   | Natural-language → Python code       | `/codegen`  |
| 🖼️ Image Generator | Stable Diffusion text-to-image       | `/imagegen` |

---

## Screenshots

🧠 **Home Dashboard**
<br><br> <img src="https://github.com/hitha-cse513/Local-GenAI-Suite/blob/main/screenshots/Screenshot 2.png" width="700">

💬 **Chatbot Interface** 
<br><br> <img src="https://github.com/hitha-cse513/Local-GenAI-Suite/blob/main/screenshots/Screenshot 6.png" width="700">

📄 **PDF Q&A Assistant**
<br><br> <img src="https://github.com/hitha-cse513/Local-GenAI-Suite/blob/main/screenshots/Screenshot 8.png" width="700">

💻 **Code Generator** 
<br><br> <img src="https://github.com/hitha-cse513/Local-GenAI-Suite/blob/main/screenshots/Screenshot 9.png" width="700">

---



## Notes

* **GPU Recommended:** Stable Diffusion requires CUDA or Apple MPS for image generation.
* **FAISS** will fall back to CPU if no GPU is available.
* You can easily integrate new tools — just create a new `.html` file and add a route in `app.py`.

---

## Credits

* 🧠 [Ollama](https://ollama.ai) — Local LLM serving framework
* 🔗 [LangChain](https://www.langchain.com) — Modular AI pipeline framework
* 🤗 [Hugging Face](https://huggingface.co) — Transformers & Diffusers
* 🎨 [Stability AI](https://stability.ai) — Stable Diffusion model
* ⚙️ [Flask](https://flask.palletsprojects.com) — Lightweight web framework

---

## License

This project is released under the **MIT License** — free for personal and commercial use with attribution.



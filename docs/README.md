# AI Prompt Analyzer

Analyzes and optimizes user prompts to improve LLM responses. Supports text and voice input.

## 🛠 Tech Stack (100% Free, No API Keys)

- **LLM**: [Ollama](https://ollama.com) running `gemma3:1b` locally (~815 MB)
- **Voice**: `faster-whisper` with `tiny` model (~75 MB, CPU only)
- **Backend**: FastAPI + Pydantic
- **Frontend**: Plain HTML/JS (no framework needed)

## 📦 Prerequisites

1. Python 3.10+
2. [Ollama](https://ollama.com/download) installed

## 🚀 Setup

### 1. Clone and install

```bash
git clone <repo>
cd ai-prompt-analyzer
pip install -r requirements.txt
```

### 2. Start Ollama and pull the model

```bash
ollama serve          # start Ollama in background
ollama pull gemma3:1b # ~815 MB download (one time)
```

> 💡 **Low RAM?** Use `gemma3:1b` (needs ~1.5 GB RAM).  
> **More space?** Use `gemma3:4b` for better quality (~2.5 GB).  
> Change `MODEL_NAME` in `src/llm_client.py`.

### 3. Start the backend

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the frontend

Open `frontend/index.html` in your browser.  
Or serve it: `python -m http.server 3000 --directory frontend`

### 5. Run tests

```bash
pytest
```

## 📡 API Endpoints

| Method | Path                 | Description              |
| ------ | -------------------- | ------------------------ |
| GET    | `/`                  | Health check             |
| POST   | `/analyze`           | Analyze text prompt      |
| POST   | `/analyze/voice`     | Analyze voice prompt     |
| GET    | `/history/{user_id}` | Get conversation history |
| DELETE | `/history/{user_id}` | Clear history            |

Interactive docs: http://localhost:8000/docs

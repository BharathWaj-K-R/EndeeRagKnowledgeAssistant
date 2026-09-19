# AI Knowledge Assistant

A small RAG application for uploading PDF/TXT documents and asking questions about their contents.

The backend uses FastAPI. Documents are split into chunks, embedded, stored in Chroma, and retrieved for each question. The Streamlit frontend talks to the backend over HTTP. There is no authentication layer in the current application.

## How it works
1. Upload a PDF or TXT file.
2. The backend extracts its text.
3. Text is split into overlapping chunks.
4. Chunks are embedded and stored in the local Chroma collection.
5. A question is embedded and the most relevant chunks are retrieved.
6. The selected answer mode generates the response:
   - `openai`: uses the configured OpenAI chat model.
   - `extractive`: returns the best retrieved passage without an LLM.
7. Retrieved sources are shown with the answer.

## Project layout

```text
app/
  main.py
  routes/
    chat.py
    health.py
    upload.py
  services/
    chat_service.py
    document_service.py
frontend/
  streamlit_app.py
rag/
  generator.py
  ingest.py
  retriever.py
utils/
  config.py
  document_loader.py
  endee_vector_store.py
  memory.py
.env.example
render.yaml
requirements.txt
```

## Local setup
Create and activate a virtual environment:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```
Install dependencies:
```powershell
pip install -r requirements.txt
```
Create the environment file:
```powershell
copy .env.example .env
```
The extractive mode does not need an OpenAI key. Add `OPENAI_API_KEY` only when using the OpenAI answer mode.
Start the backend:
```powershell
uvicorn app.main:app --reload
```
In another terminal, start Streamlit:
```powershell
streamlit run frontend/streamlit_app.py
```
Open the Streamlit URL shown in the terminal.

## Configuration
- `OPENAI_API_KEY`: required for OpenAI generation or OpenAI embeddings.
- `OPENAI_MODEL`: chat model name, default `gpt-4o-mini`.
- `EMBEDDING_PROVIDER`: `sentence_transformers`, `openai`, or `simple`.
- `EMBEDDING_MODEL_NAME`: SentenceTransformers model, default `all-MiniLM-L6-v2`.
- `OPENAI_EMBEDDING_MODEL`: OpenAI embedding model when that provider is selected.
- `CHUNK_SIZE` and `CHUNK_OVERLAP`: document chunking controls.
- `SIMILARITY_THRESHOLD`: minimum retrieval relevance score.
The default embedding provider is local SentenceTransformers. `simple` is a deterministic low-resource fallback for development, not a semantic embedding model.

## API
- `GET /health`
- `GET /api/models`
- `POST /api/upload`
- `POST /api/chat`
Uploads are indexed before `/api/upload` returns success. This avoids a race where the UI could send a question before indexing finishes. Uploads are limited to 10 MB and only PDF/TXT extensions are accepted.

## Storage
By default, uploaded temporary files are removed after indexing. Chroma data is stored under `data/chroma`.
The local filesystem is suitable for development and simple demos. Chat history is kept in process memory and is lost when the backend restarts. Ephemeral hosting can also lose the Chroma database when the service is rebuilt or restarted.

## Deployment
`render.yaml` defines one FastAPI service and one Streamlit service.
Set `OPENAI_API_KEY` in the backend service if OpenAI generation or embeddings are required. The extractive answer mode can be used without an OpenAI key.
For a hosted deployment, persistent storage is required if indexed documents must survive service restarts.

"""Streamlit frontend for interacting with the RAG backend."""

import os
import uuid

import requests
import streamlit as st


REQUEST_TIMEOUT_SECONDS = 300


def extract_error_message(error: requests.RequestException) -> str:
    """Return the most helpful backend error message available."""
    response = getattr(error, "response", None)
    if response is not None:
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("detail"):
                return str(payload["detail"])
        except ValueError:
            pass
    return str(error)


def get_backend_url() -> str:
    """Resolve the backend URL for local runs or hosted deployments."""
    direct_url = os.getenv("BACKEND_URL")
    if direct_url:
        return direct_url.rstrip("/")

    hostport = os.getenv("BACKEND_HOSTPORT")
    if hostport:
        return f"http://{hostport}"

    return "http://localhost:8000"


def fetch_models() -> list[dict]:
    """Fetch available model choices from the backend."""
    try:
        response = requests.get(f"{get_backend_url()}/api/models", timeout=30)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("models", [])
        if not models:
            raise ValueError("Backend returned no model choices.")
        return models
    except (requests.RequestException, ValueError):
        return [
            {"label": "OpenAI Chat Model", "value": "openai"},
            {"label": "Local Extractive Fallback", "value": "extractive"},
        ]


def upload_file(uploaded_file) -> dict:
    """Send the selected file to the backend upload endpoint."""
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    response = requests.post(
        f"{get_backend_url()}/api/upload",
        files=files,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def ask_question(question: str, session_id: str, model_name: str) -> dict:
    """Send a chat request to the backend and return the response payload."""
    payload = {
        "question": question,
        "session_id": session_id,
        "model_name": model_name,
        "top_k": 4,
    }
    response = requests.post(
        f"{get_backend_url()}/api/chat",
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def initialize_state() -> None:
    """Prepare Streamlit session state values used by the app."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_uploaded_file" not in st.session_state:
        st.session_state.last_uploaded_file = None


def render_sidebar(models: list[dict]) -> str:
    """Render sidebar controls and return the selected generator value."""
    st.sidebar.header("Settings")
    model_labels = {
        item["label"]: item["value"]
        for item in models
        if item.get("label") and item.get("value")
    }
    if not model_labels:
        st.sidebar.error("No answer modes are available.")
        return "extractive"

    selection = st.sidebar.selectbox("Choose answer mode", list(model_labels.keys()))
    st.sidebar.caption(
        "Use the OpenAI mode for better answers, or the extractive mode for offline and low-RAM testing."
    )
    return model_labels[selection]


def render_chat_history() -> None:
    """Display previous messages stored in Streamlit session state."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def main() -> None:
    """Run the Streamlit user interface."""
    st.set_page_config(page_title="AI Knowledge Assistant", layout="wide")
    initialize_state()
    models = fetch_models()

    st.title("AI Knowledge Assistant")
    st.write(
        "Upload PDF or TXT files, index them, and ask questions with a retrieval-augmented generation pipeline."
    )

    selected_model = render_sidebar(models)
    st.caption(
        "The first request can take longer while the backend loads the embedding model."
    )
    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])

    if uploaded_file is not None and st.session_state.last_uploaded_file != uploaded_file.name:
        try:
            result = upload_file(uploaded_file)
            st.session_state.last_uploaded_file = uploaded_file.name
            st.success(
                f"Indexed {result['filename']} with {result['chunks_indexed']} chunks into "
                f"{result['collection_name']}."
            )
        except requests.RequestException as error:
            st.error(f"Upload failed: {extract_error_message(error)}")
    elif uploaded_file is None:
        st.session_state.last_uploaded_file = None

    render_chat_history()
    question = st.chat_input("Ask a question about your documents")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        try:
            result = ask_question(
                question=question,
                session_id=st.session_state.session_id,
                model_name=selected_model,
            )
            answer = result["answer"]
            sources = result.get("sources", [])

            source_text = "\n".join(
                f"- {source.get('source', 'unknown')} (chunk {source.get('chunk_id', '?')}): "
                f"{source.get('preview', '')}"
                for source in sources
            )
            final_message = answer
            if source_text:
                final_message += f"\n\n**Sources**\n{source_text}"

            st.session_state.messages.append({"role": "assistant", "content": final_message})
            with st.chat_message("assistant"):
                st.markdown(final_message)
        except requests.RequestException as error:
            st.error(f"Question failed: {extract_error_message(error)}")


if __name__ == "__main__":
    main()

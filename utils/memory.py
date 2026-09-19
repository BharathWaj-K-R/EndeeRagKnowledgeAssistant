"""Simple in-memory chat history store."""

from collections import defaultdict


class MemoryStore:
    """Store a small amount of chat history per session."""

    MAX_MESSAGES = 20

    def __init__(self) -> None:
        """Initialize the internal session dictionary."""
        self._sessions: dict[str, list[dict]] = defaultdict(list)

    def get_history(self, session_id: str) -> list[dict]:
        """Return the saved conversation for a session."""
        return list(self._sessions[session_id])

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to a session's conversation history."""
        self._sessions[session_id].append({"role": role, "content": content})
        self._sessions[session_id] = self._sessions[session_id][-self.MAX_MESSAGES:]


memory_store = MemoryStore()

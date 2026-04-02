"""
Conversation Memory Management
- Stores per-user message history
- Limits context window to last N messages
- Summarizes old context when limit is exceeded
"""
from typing import List, Dict
from src.models import Message, MessageRole

MAX_HISTORY = 20          # max messages per user stored
CONTEXT_WINDOW = 10       # max messages sent to LLM


class ConversationMemory:
    def __init__(self):
        self._store: Dict[str, List[Message]] = {}

    def add_message(self, user_id: str, role: str, content: str):
        if user_id not in self._store:
            self._store[user_id] = []

        self._store[user_id].append(
            Message(role=MessageRole(role), content=content)
        )

        # Keep only the last MAX_HISTORY messages
        if len(self._store[user_id]) > MAX_HISTORY:
            self._store[user_id] = self._store[user_id][-MAX_HISTORY:]

    def get_history(self, user_id: str) -> List[Message]:
        """Return the most recent messages (within context window limit)."""
        messages = self._store.get(user_id, [])
        return messages[-CONTEXT_WINDOW:]

    def clear_history(self, user_id: str):
        self._store.pop(user_id, None)

    def get_summary(self, user_id: str) -> str:
        """Simple text summary of recent history for LLM context."""
        history = self.get_history(user_id)
        if not history:
            return ""
        return "\n".join(f"{m.role}: {m.content[:200]}" for m in history)

    def all_users(self) -> List[str]:
        return list(self._store.keys())
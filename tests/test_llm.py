"""Tests for LLM client parsing and caching logic."""
import pytest
from unittest.mock import patch, MagicMock
from src.llm_client import _parse_response, _clean_json, _build_cot_prompt
from src.models import Message, MessageRole


def test_parse_valid_json():
    raw = '{"issues": ["vague"], "suggestions": ["add context"], "improved_prompt": "Better prompt", "reasoning_steps": ["step1"]}'
    result = _parse_response(raw, "original")
    assert result.improved_prompt == "Better prompt"
    assert result.issues == ["vague"]


def test_parse_json_with_markdown_fences():
    raw = '```json\n{"issues": [], "suggestions": [], "improved_prompt": "Clean prompt", "reasoning_steps": []}\n```'
    result = _parse_response(raw, "original")
    assert result.improved_prompt == "Clean prompt"


def test_parse_invalid_json_uses_fallback():
    result = _parse_response("this is not json at all!!!", "my original prompt")
    assert result.improved_prompt == "my original prompt"
    assert len(result.issues) > 0


def test_clean_json_strips_fences():
    text = "```json\n{\"key\": \"val\"}\n```"
    cleaned = _clean_json(text)
    assert cleaned == '{"key": "val"}'


def test_cot_prompt_includes_user_prompt():
    prompt = _build_cot_prompt("Tell me about cats", [])
    assert "cats" in prompt
    assert "Chain-of-Thought" in prompt or "step" in prompt.lower()


def test_cot_prompt_includes_history():
    history = [
        Message(role=MessageRole.user, content="Hello"),
        Message(role=MessageRole.assistant, content="Hi there"),
    ]
    prompt = _build_cot_prompt("Follow up", history)
    assert "Hello" in prompt or "Hi there" in prompt
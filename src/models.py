"""
Pydantic schemas for structured input/output.
Uses OpenAI-compatible structured output pattern.
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from enum import Enum


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    system = "system"


class Message(BaseModel):
    role: MessageRole
    content: str


# --- Request Models ---

class AnalyzeRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=3000, description="The prompt to analyze")
    user_id: str = Field(default="default", description="User ID for conversation memory")

    @field_validator("prompt")
    @classmethod
    def strip_prompt(cls, v):
        return v.strip()


# --- Internal Models (used by LLM client) ---

class LLMAnalysisResult(BaseModel):
    """Structured output from the LLM - mirrors what the model returns."""
    issues: List[str] = Field(default_factory=list, description="Problems found in the prompt")
    suggestions: List[str] = Field(default_factory=list, description="How to improve")
    improved_prompt: str = Field(..., description="The enhanced version of the prompt")
    reasoning_steps: List[str] = Field(default_factory=list, description="Chain-of-Thought steps")


class ValidationResult(BaseModel):
    is_valid: bool
    error_message: Optional[str] = None


class InjectionResult(BaseModel):
    is_injection: bool
    details: List[str] = Field(default_factory=list)


# --- Response Models ---

class AnalyzeResponse(BaseModel):
    is_injection: bool = Field(..., description="Whether prompt injection was detected")
    injection_details: List[str] = Field(default_factory=list, description="What triggered injection detection")
    issues: List[str] = Field(default_factory=list, description="Issues found in the original prompt")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")
    improved_prompt: str = Field(..., description="The enhanced prompt")
    reasoning_steps: List[str] = Field(default_factory=list, description="CoT reasoning steps")
    context_used: bool = Field(default=False, description="Whether conversation history was used")


class VoiceAnalyzeResponse(AnalyzeResponse):
    transcribed_text: str = Field(..., description="Text transcribed from audio")


class HealthResponse(BaseModel):
    status: str
    message: str


class ConversationHistoryResponse(BaseModel):
    user_id: str
    messages: List[Message]
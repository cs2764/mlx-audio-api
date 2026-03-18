from pydantic import BaseModel, Field, field_validator
from typing import Optional


class ReferenceAudio(BaseModel):
    audio: str  # Base64 encoded audio data
    text: str = ""


class TTSRequest(BaseModel):
    text: str
    chunk_length: int = Field(default=200, ge=100, le=300)
    format: str = Field(default="wav", pattern="^(wav|mp3|pcm)$")
    references: list[ReferenceAudio] = Field(default_factory=list)
    reference_id: Optional[str] = None
    seed: Optional[int] = Field(default=42)
    use_memory_cache: str = Field(default="off", pattern="^(on|off)$")
    normalize: bool = True
    streaming: bool = False
    max_new_tokens: int = Field(default=1024, ge=0)
    top_p: float = Field(default=0.7, ge=0.1, le=1.0)
    top_k: int = Field(default=30, ge=0, le=1000)
    repetition_penalty: float = Field(default=1.1, ge=0.9, le=2.0)
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    # Ming Omni / flow-matching model params
    cfg_scale: Optional[float] = Field(default=2.0, ge=0.5, le=10.0)
    flow_steps: Optional[int] = Field(default=10, ge=1, le=100)
    sigma: Optional[float] = Field(default=0.25, ge=0.0, le=1.0)
    # Qwen3 VoiceDesign: voice style description (e.g. 'A cheerful young female voice')
    instruct: Optional[str] = None

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, v: str) -> str:
        # Reject text that contains no printable non-whitespace characters.
        # This covers: empty string, whitespace-only, and control-character-only strings.
        import unicodedata
        if not any(
            not unicodedata.category(ch).startswith(("Z", "C"))
            for ch in v
        ):
            raise ValueError("text must not be empty or whitespace-only")
        return v


class HealthResponse(BaseModel):
    status: str = "ok"


class ErrorResponse(BaseModel):
    success: bool = False
    message: str


class ReferenceAddResponse(BaseModel):
    success: bool = True
    message: str
    reference_id: str


class ReferenceListResponse(BaseModel):
    success: bool = True
    reference_ids: list[str]
    message: str


class ReferenceDeleteRequest(BaseModel):
    reference_id: str


class ReferenceUpdateRequest(BaseModel):
    old_reference_id: str
    new_reference_id: str


class QueueBusyResponse(BaseModel):
    status: str = "busy"
    message: str

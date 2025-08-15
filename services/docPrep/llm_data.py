from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class LLMData:
    run_id: str
    model: str
    created_at: str
    prompt_version: str               # e.g. git commit/tag of the prompt file
    system_prompt: str
    user_input: str                   # raw TOP text (or reference to it)
    input_hash: str                   # hash for dedupe
    params: Dict[str, Any]            # temperature, max_tokens, etc.
    output_text: Optional[str] = None
    output_json: Optional[Dict[str, Any]] = None
    latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)  # e.g. route="simple|complex"

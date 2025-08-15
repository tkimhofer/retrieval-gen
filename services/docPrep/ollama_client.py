# ollama_client.py
import time
import uuid
import requests
from typing import Any, Dict, List, Union, Optional

OllamaInput = Union[str, List[Dict[str, str]]]

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    class responses:
        pass

    def _post(self, path: str, json: Dict[str, Any], stream: bool = False):
        url = f"{self.base_url}{path}"
        resp = requests.post(url, json=json, stream=stream, timeout=300)
        resp.raise_for_status()
        return resp

    def _normalize(self, *, model: str, text: str) -> Dict[str, Any]:
        return {
            "id": f"resp_{uuid.uuid4().hex}",
            "object": "response",
            "created": int(time.time()),
            "model": model,
            "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}],
            "response": text,         # convenience
            "usage": None,            # Ollama doesn't return token counts by default
        }

    def responses_create(self, *, model: str, input: OllamaInput, stream: bool = False,
                         options: Optional[Dict[str, Any]] = None) -> Any:
        """
        input: either a string (prompt) or a list of messages [{role:'user'|'assistant'|'system', content:'...'}]
        options: forwarded to Ollama (e.g., {"temperature": 0.2})
        """
        options = options or {}

        # Chat-style input -> /api/chat
        if isinstance(input, list):
            payload = {"model": model, "messages": input, "stream": stream, **({"options": options} if options else {})}
            if stream:
                r = self._post("/api/chat", payload, stream=True)
                def gen():
                    buf = ""
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        chunk = line
                        # Each line is JSON; we only pull "message.content" if present
                        try:
                            data = requests.utils.json.loads(chunk)
                            msg = data.get("message", {}).get("content", "")
                            buf += msg
                            yield {"type": "response.delta", "delta": msg}
                        except Exception:
                            pass
                    yield {"type": "response.completed", "response": self._normalize(model=model, text=buf)}
                return gen()
            else:
                data = self._post("/api/chat", payload).json()
                text = data.get("message", {}).get("content", "")
                return self._normalize(model=model, text=text)

        # String prompt -> /api/generate
        else:
            payload = {"model": model, "prompt": str(input), "stream": stream, **({"options": options} if options else {})}
            if stream:
                r = self._post("/api/generate", payload, stream=True)
                def gen():
                    buf = ""
                    for line in r.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        try:
                            data = requests.utils.json.loads(line)
                            part = data.get("response", "")
                            buf += part
                            yield {"type": "response.delta", "delta": part}
                            if data.get("done"):
                                break
                        except Exception:
                            pass
                    yield {"type": "response.completed", "response": self._normalize(model=model, text=buf)}
                return gen()
            else:
                data = self._post("/api/generate", payload).json()
                text = data.get("response", "")
                return self._normalize(model=model, text=text)

# Provide OpenAI-like access path: client.responses.create(...)
class Client(OllamaClient):
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__

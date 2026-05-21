import json
from urllib.parse import urljoin
import logging
import asyncio
# from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv
load_dotenv('.env')
from typing import List

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse

LMSTUDIO_BASE_URL = os.getenv('LMSTUDIO_BASE_URL')
LMSTUDIO_ENDPOINT = "v1/chat/completions"
LMSTUDIO_SUBMIT_URL = urljoin(LMSTUDIO_BASE_URL, LMSTUDIO_ENDPOINT)
MODEL_NAME = "openai/gpt-oss-20b" # "qwen/qwen3-coder-30b"

prompt = 'How are you?'

# Ollama
json={"model": MODEL_NAME, "prompt": prompt, "stream": True}

# LM Studio
json={
    "model": MODEL_NAME,
    "messages": [
        {"role": "system", "content": "you are a friendly chatbot."},
        {"role": "user", "content": prompt},
    ],
    "stream": True,
}





async def iterator():
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=None, connect=5.0, read=None, write=10.0, pool=10.0)
    ) as client:
        async with client.stream(
            "POST",
            LMSTUDIO_SUBMIT_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Du bist ein Assistent, der Fragen zu Ratssitzungen "
                            "der Stadt Duisburg beantwortet. Antworte ausschließlich "
                            "auf Deutsch und nur mit Hilfe des bereitgestellten Kontexts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "temperature": 0.2,
                "stream": True,
            },
        ) as r:
            if r.status_code != 200:
                err = await r.aread()
                msg = err.decode(errors="ignore")[:2000]
                yield (json.dumps({"error": f"LM Studio {r.status_code}: {msg}"}) + "\n").encode("utf-8")
                return

            async for line in r.aiter_lines():
                if not line:
                    continue

                # LM Studio/OpenAI-compatible streaming uses SSE:
                # data: {"choices":[{"delta":{"content":"..."}}]}
                if not line.startswith("data: "):
                    continue

                data = line.removeprefix("data: ").strip()

                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                    token = (
                        chunk.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content", "")
                    )
                except Exception:
                    continue

                if token:
                    # keep your old NDJSON output shape for Chainlit client
                    yield (json.dumps({"response": token}) + "\n").encode("utf-8")
import json
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

from services.retrieval.Retriever import retr


# api_key = os.getenv('GPT_API_KEY')
# client = AsyncOpenAI(api_key=api_key)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mixtral"

app = FastAPI()

# ---- Retrieval (run sync work off-thread if needed) ----
def _retrieve_sync(query: str) -> List[str]:
    retr.query_db(query)
    out = retr.rerank_search_results(return_n=2)
    # adapt if your tuple layout differs
    return [x[3] for x in out]

async def retrieve(query: str) -> List[str]:
    return await asyncio.to_thread(_retrieve_sync, query)

def build_prompt(user_msg: str, context_docs: List[str]) -> str:
    context_text = "\n\n".join(context_docs) if context_docs else "(Kein Kontext gefunden.)"
    return (
        "Du bist ein Assistent der Fragen bezüglich Ratssitzungen der Stadt Duisburg beantwortet. Dazu fasst du Kontext zusammenfasst. \n\n"
        f"Kontext:\n{context_text}\n\n"
        f"Frage: {user_msg}\n\n"
        "Beantworte die Frage nur mit Hilfe des Kontexts. Der Kontext stammt aus Dokumenten von Ratssitzungen der Stadt Duisburg. "
        "Antworte in einfacher Sprache und ausschließlich in Deutsch."
    )

def build_prompt_openai(user_msg: str, context_docs: List[str]) -> str:
    context_text = "\n\n".join(context_docs) if context_docs else "(Kein Kontext gefunden.)"
    return (
        "Du bist ein Assistent der Fragen bezüglich Ratssitzungen der Stadt Duisburg beantwortet. Dazu fasst du Kontext zusammenfasst. \n\n"
        f"Kontext:\n{context_text}\n\n"
        f"Frage: {user_msg}\n\n"
        "Beantworte die Frage nur mit Hilfe des Kontexts. Der Kontext stammt aus Dokumenten von Ratssitzungen der Stadt Duisburg. "
        "Antworte in einfacher Sprache und ausschließlich in Deutsch."
    )

@app.post("/api/rag")
async def rag_stream(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    user_msg = (body.get("prompt") or "").strip()
    user_id = body.get("user_id")

    if not user_msg:
        raise HTTPException(status_code=400, detail="Field 'prompt' is required")

    logger.info(f"USER={user_id} PROMPT={user_msg}")

    # 1) Retrieve context (non-blocking for other requests)
    context_docs = await retrieve(user_msg)
    prompt = build_prompt(user_msg, context_docs)

    # 2) Stream to Ollama and re-stream out (NDJSON lines)
    async def iterator():
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout=None, connect=5.0, read=None, write=10.0, pool=10.0)) as client:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={"model": MODEL_NAME, "prompt": prompt, "stream": True},
            ) as r:
                if r.status_code != 200:
                    err = await r.aread()
                    msg = err.decode(errors="ignore")[:2000]
                    yield (json.dumps({"error": f"Ollama {r.status_code}: {msg}"}) + "\n").encode("utf-8")
                    return

                async for line in r.aiter_lines():
                    if not line:
                        continue
                    # Expect NDJSON from Ollama: {"response": "...", ...}
                    # Forward as-is so the Chainlit client can parse `.get("response")`
                    yield (line + "\n").encode("utf-8")

    return StreamingResponse(
        iterator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )

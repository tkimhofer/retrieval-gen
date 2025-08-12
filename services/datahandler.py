from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional
import hashlib, json, os, uuid
import datetime as dt
import re
from textwrap import dedent
from git import Repo
import time, os

from llm_data import LLMData


class LLMRun:

    def __init__(self, client):

        self.client = client
        self.prompt = self.import_prompt()


    def run(self, model, top_id, meeting_date, top_text, pars={"temperature": 0.2}):

        meta = {'meeting_date': meeting_date, 'top_id': top_id}

        version_info = self.last_commit_gitpython(repo_path=".")

        query = LLMRun(
            run_id=str(uuid.uuid4()),
            model=model,
            created_at=self.now_iso(),
            prompt_version=version_info["message"],  # tie to a git commit/tag if you can
            system_prompt=self.prompt,
            user_input=top_text,
            input_hash=self.sha256(top_text),
            params=pars,
            meta = meta
        )

        t0 = time.time()
        model_response = self.model_response(model,)
        tdiff_ms = int((time.time() - t0) * 1000)

        query.output_text  = getattr(model_response, "output_text", None)
        query.latency_ms = tdiff_ms

        usage = getattr(model_response, "usage", None)
        if usage:
            query.input_tokens = getattr(usage, "input_tokens", None)
            query.output_tokens = getattr(usage, "output_tokens", None)

        # write data to file
        file_id = "_".join(query.meta.values()) + "_" + query.created_at + '.jsonl'
        file_path = os.path.join('model_response', file_id)
        self.log_run_to_jsonl(data=query, path=file_path)

    @staticmethod
    def last_commit_gitpython(repo_path="."):
        repo = Repo(repo_path, search_parent_directories=True)
        c = repo.head.commit
        return {
            "hash": c.hexsha,
            "subject": c.message.splitlines()[0],
            "message": c.message,
            "author": f"{c.author.name} <{c.author.email}>",
            "date": c.committed_datetime.isoformat(),
        }

    @staticmethod
    def import_prompt(path: str = 'model_prompt.txt'):
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
            clean = dedent(raw)
            clean = re.sub(r'\n{3,}', '\n\n', clean)
        return clean.strip()

    @staticmethod
    def now_iso() -> str:
        return dt.now(dt.timezone.utc).isoformat()

    @staticmethod
    def sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def log_run_to_jsonl(data: LLMData, path: str = "llm_runs.jsonl") -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(data), ensure_ascii=False) + "\n")


    def model_response(self, model: List[OllamaClient, Client], prompt_instructions: str, top_text: str):

        model_input = [
            {"role": "system", "content": prompt_instructions},
            {"role": "user", "content": f"INPUT (TOP-Text): {top_text}"}
        ]

        model_output = self.client.responses.create(
            model=model,
            input=model_input,
        )

        return model_output




"""OpenAI-compatible chat client with a content-addressed response cache.

Every request and response is stored under a cache directory keyed by the SHA-256 of
the canonical request, so experiments that used a live model can be re-run and audited
offline from the recorded responses. Model identifiers, token counts, and latency are logged.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .types import canonical_json, digest

ENDPOINTS = {
    "clover": {"base_url": os.getenv("CLOVER_BASE_URL", "http://ken-mac.local:8000/v1"), "model": "clover-35b-a3b"},
    "qwen4b": {"base_url": os.getenv("CLOVER_BASE_URL", "http://ken-mac.local:8000/v1"), "model": "qwen3.5-4b:4bit"},
    "deepseek": {"base_url": os.getenv("DEEPSEEK_BASE_URL", "http://localhost:8888/v1"), "model": "deepseek"},
}


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, name, cache_dir, *, offline=False, timeout=900, max_retries=2):
        if name not in ENDPOINTS:
            raise ValueError(f"unknown endpoint {name}")
        self.name = name
        self.base_url = ENDPOINTS[name]["base_url"].rstrip("/")
        self.model = ENDPOINTS[name]["model"]
        self.cache = Path(cache_dir) / name
        self.cache.mkdir(parents=True, exist_ok=True)
        self.offline, self.timeout, self.max_retries = offline, timeout, max_retries
        self.stats = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0, "errors": 0}

    def chat(self, messages, *, temperature=0.0, max_tokens=2048, sample=0):
        """`sample` distinguishes repeated draws of an identical request at temperature > 0."""
        request = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        key = digest({"request": request, "sample": sample})
        path = self.cache / f"{key}.json"
        if path.is_file():
            self.stats["cache_hits"] += 1
            return json.loads(path.read_text(encoding="utf-8"))
        if self.offline:
            raise LLMError(f"offline and no cached response for {key}")
        body = canonical_json(request).encode()
        last = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    raw = json.loads(response.read().decode())
                break
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last = error
                self.stats["errors"] += 1
                time.sleep(2 * (attempt + 1))
        else:
            raise LLMError(f"{self.name} request failed: {last}")
        elapsed = time.perf_counter() - start
        message = raw["choices"][0]["message"]
        usage = raw.get("usage", {})
        record = {"endpoint": self.name, "model": raw.get("model", self.model), "request": request, "sample": sample,
                  "content": message.get("content") or "", "finish_reason": raw["choices"][0].get("finish_reason"),
                  "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0),
                  "seconds": round(elapsed, 3)}
        self.stats["calls"] += 1
        self.stats["prompt_tokens"] += record["prompt_tokens"]
        self.stats["completion_tokens"] += record["completion_tokens"]
        self.stats["seconds"] += elapsed
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        return record


def extract_json(text):
    """Return the first top-level JSON object in a model response, or None."""
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>")[-1]
    for fence in ("```json", "```"):
        if fence in text:
            parts = text.split(fence)
            if len(parts) > 1:
                text = parts[1].split("```")[0]
                break
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None

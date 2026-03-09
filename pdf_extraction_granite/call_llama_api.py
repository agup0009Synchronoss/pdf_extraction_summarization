"""
call_llama_api.py
-----------------
Minimal wrapper around your Ollama / llama-3 endpoint.

export LLAMA_API_KEY=xxxxx   # or hard-code below
"""
from __future__ import annotations
import json, os, requests, typing as _t

API_URL  = "https://ml-llm-api-gpu.use.eks.mcap.sip.dev.cloud.synchronoss.net/api/chat"
MODEL    = "llama3.1:8b-instruct-q8_0"
API_KEY  = os.getenv("LLAMA_API_KEY") or "mPt3X5N_dY9W"   # TODO replace!

_SYSTEM  = ("You are an AI assistant that follows instructions extremely well. "
            "Return ONLY the JSON requested by the user.")

def build_payload(prompt: str,
                  temperature: float = 0.2,
                  seed: int = 42) -> dict[str, _t.Any]:
    return {
        "model": MODEL,
        "stream": False,
        "options": {"temperature": temperature, "seed": seed},
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": prompt},
        ],
    }

def send_prompt(prompt: str,
                api_key: str | None = None) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        "X-Api-Key": api_key or API_KEY,
    }
    return requests.post(API_URL,
                         headers=headers,
                         data=json.dumps(build_payload(prompt)),
                         timeout=120)

# quick manual test ----------------------------------------------------------
if __name__ == "__main__":
    import sys, textwrap
    text = sys.argv[1] if len(sys.argv) > 1 else "Say hello."
    r = send_prompt(text)
    print(f"STATUS {r.status_code}\n" + textwrap.fill(r.text, 110))

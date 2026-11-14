from pathlib import Path
import os
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=True)

api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise SystemExit("OPENROUTER_API_KEY is missing from .env")

url = "https://openrouter.ai/api/v1/chat/completions"
payload = {
    "model": "nvidia/nemotron-3.5-super-vl:free",
    "messages": [{"role": "user", "content": "Reply with exactly: API_TEST_OK"}],
    "max_tokens": 20,
    "temperature": 0,
}

try:
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai/",
            "X-Title": "IoT Dataset API Diagnostic",
        },
        json=payload,
        timeout=60,
    )
    print(f"HTTP STATUS: {response.status_code}")
    print("\nRELEVANT RESPONSE HEADERS:")
    for name in ("retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"):
        value = response.headers.get(name)
        if value is not None:
            print(f"{name}: {value}")
    print("\nPROVIDER RESPONSE BODY:")
    print(response.text[:4000])
except requests.RequestException as exc:
    print(f"NETWORK ERROR: {type(exc).__name__}: {exc}")
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
    "messages": [
        {
            "role": "system",
            "content": "You are a concise assistant. Do not provide reasoning. Return only the requested text."
        },
        {
            "role": "user",
            "content": "Return exactly this text and nothing else: API_TEST_OK"
        },
    ],
    "max_tokens": 256,
    "temperature": 0,
}

try:
    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openrouter.ai/",
            "X-Title": "IoT Dataset Generation Diagnostic",
        },
        json=payload,
        timeout=90,
    )
    print(f"HTTP STATUS: {response.status_code}")
    print("\nPROVIDER RESPONSE BODY:")
    try:
        data = response.json()
        if response.ok:
            choice = (data.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            print("MODEL:", data.get("model"))
            print("FINISH REASON:", choice.get("finish_reason"))
            print("CONTENT:", repr(message.get("content")))
            print("REASONING:", repr(message.get("reasoning"))[:500])
            print("USAGE:", data.get("usage"))
        else:
            print(response.text[:4000])
    except ValueError:
        print(response.text[:4000])
except requests.RequestException as exc:
    print(f"NETWORK ERROR: {type(exc).__name__}: {exc}")
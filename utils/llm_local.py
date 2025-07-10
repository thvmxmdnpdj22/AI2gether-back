# utils/llm_local.py

import requests

def make_prompt(place_name, info, days_text="1박 2일"):
    return f"""
당신은 여행 컨설턴트입니다.

추천된 여행지는 {place_name.upper()}입니다.
사용자의 여행 기간은 {days_text}입니다.

- 관광지: {', '.join(info['명소'])}
- 음식: {', '.join(info['음식'])}
- 숙소: {', '.join(info['숙소'])}

위 정보를 바탕으로 사용자에게 {days_text} 여행 동선을 친근하고 따뜻한 문체로 4~6줄로 작성해주세요.
"""

def call_llama3(prompt, host="http://localhost:11434", model="llama3"):
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["response"].strip()
    except Exception as e:
        return f"[ERROR] LLM 호출 실패: {e}"

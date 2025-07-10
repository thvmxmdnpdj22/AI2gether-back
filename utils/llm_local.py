# utils/llm_local.py

import requests

def extract_names(data_list):
    if isinstance(data_list[0], dict):
        return [item["name"] for item in data_list]
    return data_list

def make_prompt(place_name, info, days_text="1박 2일"):
    sights = extract_names(info["명소"])
    foods = extract_names(info["음식"])
    hotels = extract_names(info["숙소"])

    return f"""
당신은 여행 컨설턴트입니다.

추천된 여행지는 {place_name.upper()}입니다.

- 여행 일정: {days_text}
- 관광지: {', '.join(sights)}
- 음식: {', '.join(foods)}
- 숙소: {', '.join(hotels)}

위 정보를 바탕으로 사용자에게 오전/오후 일정과 함께 관광지/음식점/숙소를 포함한 여행 동선을 따뜻한 문체로 4~6줄로 작성해주세요.
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

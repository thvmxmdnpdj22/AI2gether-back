# pages/detail_page.py
import streamlit as st
from utils.google_maps import get_coordinates
from utils.llm_local import make_prompt, call_llama3
from utils.travel_info import travel_info
import streamlit.components.v1 as components
from config import GOOGLE_MAPS_API_KEY

st.set_page_config(
    page_title="여행지 상세 정보",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 2rem;
        max-width: 900px;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

# ✅ 필요한 세션 정보가 없을 경우 홈으로 리다이렉트
if "selected_place" not in st.session_state:
    st.warning("추천 화면에서 먼저 여행지를 선택해주세요.")
    st.switch_page("app.py")

place = st.session_state.selected_place
info = travel_info.get(place, {})
days_text = st.session_state.get("selected_days", "1박 2일")

st.title(f"🌟 {place.upper()} 여행 상세 정보")

# ✅ LLM 결과 없으면 생성
if "llm_route" not in st.session_state or not st.session_state.llm_route:
    prompt = make_prompt(place, info, days_text)
    st.session_state.llm_route = call_llama3(prompt)

# ✨ 여행 루트 출력
st.subheader("🧭 추천 여행 일정")
st.write(st.session_state.llm_route)

# 🗺️ 지도 시각화
st.subheader("📍 여행지 지도 보기")

def render_google_map(location_name):
    lat, lng = get_coordinates(location_name, GOOGLE_MAPS_API_KEY)
    if lat and lng:
        map_html = f"""
        <iframe width="100%" height="300" 
        src="https://www.google.com/maps?q={lat},{lng}&hl=ko&z=14&output=embed"></iframe>
        """
        components.html(map_html, height=300)
    else:
        st.warning(f"'{location_name}'의 위치 정보를 찾을 수 없습니다.")

# ✨ 주요 장소 지도 표시
for keyword in info.get("명소", [])[:2] + info.get("음식", [])[:1] + info.get("숙소", [])[:1]:
    st.markdown(f"### 📍 {keyword}")
    render_google_map(keyword)
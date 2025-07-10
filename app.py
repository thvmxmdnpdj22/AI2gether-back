# app.py
import streamlit as st
from PIL import Image
import os
import random

from model.clip_model import load_clip_model, get_image_embedding
from utils.similarity import find_top_k_similar_images_distinct
from utils.image_loader import load_travel_images
from utils.travel_info import travel_info

# 👉 화면 설정: 중앙 정렬 + 사이드바 숨김
st.set_page_config(page_title="여행지 추천 시스템", layout="centered", initial_sidebar_state="collapsed")
st.markdown("<style>footer {visibility: hidden;}</style>", unsafe_allow_html=True)

# 모델 로딩
@st.cache_resource
def load_model():
    return load_clip_model()

model, processor = load_model()
travel_places = load_travel_images("img_data")

st.title("🌍 사진 기반 여행지 추천 시스템")

# 👉 사용자 이미지 업로드
uploaded_file = st.file_uploader("📸 여행 사진을 업로드하세요 (jpg, png)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_container_width=False, width=400)  # 적당한 크기

    user_vecs = get_image_embedding([image], model, processor)
    if user_vecs:
        top_results = find_top_k_similar_images_distinct(
            user_vecs[0], travel_places, model, processor, k=3
        )

        st.markdown("### ✨ 추천 여행지")
        cols = st.columns(3)  # 3열로 정렬

        for idx, (score, image_path, place) in enumerate(top_results):
            info = travel_info.get(place, {})
            desc = random.choice(info.get("명소", ["명소 정보 없음"]))

            # ✅ 유사도 score를 터미널에 출력
            print(f"[유사도] {place} : {score:.4f}")

            with cols[idx]:
                st.image(
                    image_path,
                    caption=f"{place.upper()} - {desc}",
                    use_container_width=False,
                    width=220  # ✅ 고정 크기
                )
                st.markdown(f"🧭 유사도: **{score:.2f}**", unsafe_allow_html=True)

                if st.button(f"📍 {place} 선택", key=f"select_{idx}"):
                    st.session_state.selected_place = place
                    st.session_state.selected_info = info
                    st.session_state.llm_route = None
                    st.switch_page("pages/detail_page.py")
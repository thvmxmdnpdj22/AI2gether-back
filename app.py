import streamlit as st
from model.clip_model import load_clip_model, get_image_embedding
from utils.similarity import find_best_match, find_most_similar_image
from utils.travel_info import travel_info
from utils.llm_local import make_prompt, call_llama3
from utils.image_loader import load_travel_images, get_representative_images
from PIL import Image
import os
from utils.coordinates import place_coordinates



# 모델 로딩
@st.cache_resource
def load_model():
    return load_clip_model()

model, processor = load_model()

# 여행지 이미지 자동 로딩
@st.cache_resource
def prepare_travel_vectors():
    travel_places = load_travel_images("img_data")
    travel_vectors = {}
    for place, image_list in travel_places.items():
        vec = get_image_embedding(image_list, model, processor)
        if vec is not None:
            travel_vectors[place] = vec
    return travel_vectors

# 이미지 데이터 불러오기
travel_vectors = prepare_travel_vectors()
travel_places = load_travel_images("img_data")
place_images = get_representative_images(travel_places)

# Streamlit UI
st.title("🌍 사진 기반 여행지 추천 시스템")

uploaded_file = st.file_uploader("당신의 여행 사진을 업로드하세요 (jpg, png)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드한 이미지", use_column_width=True)

    num_days = st.text_input("✈️ 몇 박 몇 일 여행을 계획하시나요?", placeholder="예: 1박 2일, 3박 4일")

    if num_days:  # ✅ 날짜가 입력된 경우에만 분석 진행
        with st.spinner("여행지를 분석 중입니다..."):
            user_vecs = get_image_embedding([image], model, processor)
            if user_vecs:
                best_match = find_best_match(user_vecs[0], travel_vectors)
                st.success(f"✨ 추천 여행지는: **{best_match.upper()}**")

                # 대표 이미지 출력
                candidate_imgs = travel_places.get(best_match)
                best_img_path = find_most_similar_image(user_vecs[0], candidate_imgs, model, processor)
                if best_img_path and os.path.exists(best_img_path):
                    st.image(best_img_path, caption=f"{best_match.upper()}와 가장 유사한 이미지", use_container_width=True)
                else:
                    st.warning("추천된 여행지의 유사한 이미지를 찾을 수 없습니다.")

                # LLM 설명 생성
                if best_match in travel_info:
                    info = travel_info[best_match]
                    prompt = make_prompt(best_match, info, days_text=num_days)

                    with st.spinner("✨ 로컬 AI가 여행 동선을 작성 중입니다..."):
                        result_text = call_llama3(prompt)
                        st.info("🧭 추천 여행 루트")
                        st.write(result_text)
                else:
                    st.warning("추천된 여행지에 대한 설명 데이터를 찾을 수 없습니다.")
            else:
                st.error("이미지에서 특징을 추출하지 못했습니다. 다시 시도해보세요.")
    else:
        st.info("✏️ 여행 일정을 입력해주세요. (예: 1박 2일)")
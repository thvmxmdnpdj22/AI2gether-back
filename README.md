# 여행지 추천 AI 시스템 (사진 기반)

### 다운로드 ###
https://drive.google.com/file/d/1m_XpBTS6yZ-7JXnSDBJ5vkJfb69mmC-B/view?usp=sharing

## 개요
사용자가 업로드한 여행 사진을 분석하여, 가장 유사한 인기 여행지를 추천하는 시스템입니다.  
CLIP 모델로 이미지 임베딩 후, 대표 여행지들과 유사도를 비교합니다.

## 구조

min_project/
- `app.py` : Streamlit 웹앱 진입점
- `train_clip.py`: CLIP 모델 파인튜닝 실행 스크립트
- `config.py` : 하이퍼파라미터 및 경로 설정
- `img_data/`: 여행지 이미지 (폴더별로 분류됨)

model/ : 모델 로딩 및 이미지 임베딩 처리
- `clip_model.py`  모델 로딩 및 이미지 임베딩 추출 함수

utils/: 유사도 계산 유틸리티
- `data_utils.py` : DataLoader 및 Dataset 정의
- `train_utils.py` : 학습 루프 함수 정의
- `image_loader.py` : 여행지 이미지 불러오기 및 전처리
- `similarity.py` : 벡터 간 유사도 계산
- `travel_info.py` : 각 여행지에 대한 설명 정보

## 주요 기능 흐름
1. 학습(`train_clip.py`)
- 여행지 사진들을 학습용으로 사용하여 CLIP 모델을 파인튜닝
- 결과는 `finetuned_clip/`에 저장

2. 추천(`app.py`)
- 사용자가 이미지를 업로드
- `model/clip_model.py`에서 파인튜닝된 모델 로딩
- 업로드 이미지 -> 벡터 임베딩 -> 기존 여행지들과 cosine 유사도 비교
- 가장 유사한 여행지 + 대표 이미지 추천 + 설명 출력 + 

### 사용 기술
기술 설명
- CLIP (openai/clip-vit-base-patch32) : 이미지/텍스트 임베딩 생성
- HuggingFace Transformers : CLIP 모델 및 Processor 사용
- PyTorch : 모델 파인튜닝 및 추론
- Streamlit : 사용자 인터페이스 구성
- cosine similarity : 유사도 비교 기반 추천 알고리즘

# 실행법

# 1. 패키지 설치
pip install -r requirements.txt

# 2. CLIP 모델 파인튜닝 (선택)
python train_clip.py

# 3. Streamlit 앱 실행
streamlit run app.py

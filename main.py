from model.clip_model import load_clip_model, get_image_embedding
from utils.similarity import find_best_match

def main():
    # 1. 모델 로드
    model, processor = load_clip_model()

    # 2. 여행지 이미지 로드
    travel_places = {
        "jeju": ["img_data/jeju1.jpg", "img_data/jeju2.jpg"],
        "malta": ["img_data/malta1.jpg", "img_data/malta2.jpg"],
        "boracay": ["img_data/boracay1.jpg", "img_data/boracay2.jpg"],
        "maldives": ["img_data/maldives1.jpg", "img_data/maldives2.jpg"],
        "hawaii": ["img_data/hawaii1.jpg", "img_data/hawaii2.jpg"]
    }

    travel_vectors = {}
    for place, image_list in travel_places.items():
        vec = get_image_embedding(image_list, model, processor)
        if vec is not None:
            travel_vectors[place] = vec
        else:
            print(f" {place} 여행지 벡터 생성 실패 — 이미지 확인 필요")


    # 3. 사용자 이미지 임베딩
       # 3. 사용자 이미지 임베딩
    user_image = "user_input/test_user.jpg"
    user_result = get_image_embedding([user_image], model, processor)
    
    if not user_result:
        print(" 사용자 이미지를 불러오지 못했습니다.")
        return
    user_vector = user_result[0]

    # 4. 유사한 여행지 추천
    best_match = find_best_match(user_vector, travel_vectors)

    print(f"\n 추천 여행지: {best_match.upper()}")

if __name__ == "__main__":
    main()

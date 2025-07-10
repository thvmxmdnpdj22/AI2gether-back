from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import heapq
import numpy as np


def find_best_match(user_vector, travel_vectors: dict):
    best = None
    best_score = -1
    for place, vec in travel_vectors.items():
        score = cosine_similarity(user_vector, vec, dim=0).item()
        print(f"🔍 {place}: {score:.4f}")
        if score > best_score:
            best_score = score
            best = place
    return best.lower()  # ← 🔥 이 한 줄만 추가


def find_top_k_unique_places(user_vec, travel_places, model, processor, top_k=3):


    scored_images = []

    for place_name, image_paths in travel_places.items():
        best_score = -1
        best_image_path = None

        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt", padding=True)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            similarity = torch.nn.functional.cosine_similarity(
                user_vec.unsqueeze(0), image_features
            ).item()

            if similarity > best_score:
                best_score = similarity
                best_image_path = image_path

        if best_image_path:
            scored_images.append((best_score, best_image_path, place_name))

    # 유사도 상위 top_k개 선택
    top_k_images = heapq.nlargest(top_k, scored_images, key=lambda x: x[0])
    return top_k_images



def find_most_similar_image(user_embedding, candidate_images, model, processor):
    max_sim = -1
    best_image_path = None

    for img_path in candidate_images:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            image_emb = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

        sim = torch.cosine_similarity(user_embedding, image_emb).item()

        if sim > max_sim:
            max_sim = sim
            best_image_path = img_path

    return best_image_path


def find_top_k_similar_images_distinct(user_vector, travel_places, model, processor, k=3):
    import heapq
    from PIL import Image
    import torch
    from torch.nn.functional import cosine_similarity

    scored_images = []

    for place_name, image_paths in travel_places.items():
        best_score = -1
        best_image_path = None

        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(model.device)
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            sim = cosine_similarity(user_vector.unsqueeze(0), image_features).item()
            if sim > best_score:
                best_score = sim
                best_image_path = image_path

        if best_image_path:
            scored_images.append((best_score, best_image_path, place_name))

    # 유사도 상위 k개 장소를 뽑되, 서로 다른 장소여야 함
    top_k_images = heapq.nlargest(k, scored_images, key=lambda x: x[0])
    return top_k_images



from torch.nn.functional import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image



def find_best_match(user_vector, travel_vectors: dict):
    best = None
    best_score = -1
    for place, vec in travel_vectors.items():
        score = cosine_similarity(user_vector, vec, dim=0).item()
        print(f"🔍 {place}: {score:.4f}")
        if score > best_score:
            best_score = score
            best = place
    return best




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
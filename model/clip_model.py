from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch


def load_clip_model():
    model_path = "finetuned_clip"  # ✅ 우리가 저장한 모델 경로
    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)
    return model, processor

def get_image_embedding(image_inputs, model, processor):
    vectors = []
    for item in image_inputs:
        try:
            if isinstance(item, str):
                image = Image.open(item)
            else:
                image = item  # 이미 PIL 객체일 경우
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                features = model.get_image_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            vectors.append(features[0])
        except Exception as e:
            print(f"❌ 이미지 처리 실패: {item}, 오류: {e}")
    if not vectors:
        return None
    return torch.stack(vectors).mean(dim=0) if len(vectors) > 1 else vectors

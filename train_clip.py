# train_clip.py
from transformers import CLIPProcessor, CLIPModel
from utils.data_utils import get_dataloader
from utils.train_utils import train_clip
from config import CONFIG

print("📦 모델 로딩 중...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("🔧 데이터 로딩 중...")
dataloader, class_names = get_dataloader(CONFIG["image_root"], CONFIG["batch_size"])

print(f"클래스 수: {len(class_names)}")

print("🚀 파인튜닝 시작!")
train_clip(model, processor, dataloader, class_names, CONFIG)

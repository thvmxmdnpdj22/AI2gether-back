# utils/data_utils.py
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from PIL import Image
import os
import numpy as np
import torch

# 전역 processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class CLIPDataset(ImageFolder):
    def __init__(self, root, used_images_path=None):
        super().__init__(root)
        self.used_images = self.get_used_image_paths(used_images_path)
        self.samples = [s for s in self.samples if s[0] not in self.used_images]

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            # 1. PIL로 불러오기
            image = Image.open(path).convert("RGB")

            # 2. Processor로 전처리
            inputs = processor(images=image, return_tensors="pt")

            # 3. pixel_values는 float32 텐서이며, 값은 [-1, 1] 범위 → 문제가 되는 부분
            pixel_values = inputs["pixel_values"].squeeze(0)  # [3, H, W]

            # 4. 안정성을 위해 클램핑 (혹시 모를 정규화 오버슈팅 방지)
            pixel_values = pixel_values.clamp(0, 1)

            return pixel_values, label

        except Exception as e:
            print(f"[ERROR] 이미지 로딩 실패: {path}, 오류: {e}")
            return self.__getitem__((index + 1) % len(self.samples))


    def get_used_image_paths(self, path):
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f.readlines())
        return set()

def get_dataloader(image_root, batch_size=8, shuffle=True, used_images_path=None):
    dataset = CLIPDataset(root=image_root, used_images_path=used_images_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, dataset.classes

import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


def train_clip(model, processor, dataloader, class_names, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()

    # ✅ 전체 클래스 이름을 기준으로 텍스트 입력 생성
    text_inputs = processor(
        text=class_names,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    for epoch in range(config["num_epochs"]):
        total_loss = 0

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            images = images.to(device)
            labels = labels.to(device)

            # ✅ 이미지 배치에 대해서만 전처리
            image_inputs = processor(
                images=images,
                return_tensors="pt",
                do_rescale=False
            ).to(device)

            # ✅ forward
            outputs = model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                pixel_values=image_inputs.pixel_values
            )
            logits = outputs.logits_per_image  # shape: [batch_size, num_classes]

            loss = F.cross_entropy(logits, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} 평균 loss: {total_loss:.4f}")

    # ✅ 모델 저장
    model.save_pretrained(config["save_path"])
    processor.save_pretrained(config["save_path"])
    print(f"\n✅ 모델 저장 완료: {config['save_path']}")

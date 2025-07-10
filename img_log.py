#이미지 로그 used_images.txt에 저장
import os

image_dir = "img_data"  # 혹은 네가 사용한 폴더 경로
used_list = []

for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith((".jpg", ".png")):
            full_path = os.path.join(root, file)
            used_list.append(full_path)

# 파일로 저장
with open("used_images.txt", "w", encoding="utf-8") as f:
    for path in used_list:
        f.write(path + "\n")
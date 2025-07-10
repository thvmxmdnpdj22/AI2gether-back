# utils/image_loader.py

import os
from collections import defaultdict

def load_travel_images(base_path="img_data"):
    """
    여행지별 이미지 리스트를 자동으로 구성해 딕셔너리로 반환
    {
        "방콕": ["img_data/방콕/1.jpg", "img_data/방콕/2.jpg"],
        ...
    }
    """
    travel_places = defaultdict(list)

    for region in os.listdir(base_path):
        region_path = os.path.join(base_path, region)
        if not os.path.isdir(region_path):
            continue
        for file in os.listdir(region_path):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                relative_path = os.path.join(base_path, region, file)
                travel_places[region].append(relative_path)

    return dict(travel_places)


def get_representative_images(travel_places):
    """
    여행지별 대표 이미지를 뽑아 딕셔너리로 반환
    {
        "방콕": "img_data/방콕/1.jpg",W
        ...
    }
    """
    return {region: images[0] for region, images in travel_places.items() if images}

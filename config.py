# config.py
CONFIG = {
    "image_root": "img_data",
    "batch_size": 8,
    "lr": 5e-6,
    "num_epochs": 5,
    "save_path": "finetuned_clip"
}


import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
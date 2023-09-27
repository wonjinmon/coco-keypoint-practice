import json

from tqdm import tqdm
from PIL import Image, ImageDraw

import numpy as np


with open("infant-dataset/annotations.json", "r") as f:
    data = json.load(f)

# data["annotations"]["categories"]
cats = data["categories"][0]["keypoints"]


for idx, img_info in enumerate(data["images"]):
    img_path = "infant-dataset/" + img_info["file_name"]

    image = Image.open(img_path)

    keypoints = data["annotations"][idx]["keypoints"]
    keypoints = np.array(keypoints).reshape(-1, 3)

    draw = ImageDraw.Draw(image)
    for i, kp in tqdm(enumerate(keypoints)):
        x, y, _ = kp
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 255), width=2)
        draw.text((x - 5, y - 5), str(i))
    image.save(f'./keypoints/{img_info["id"]}_{str(idx+1)}.jpg')

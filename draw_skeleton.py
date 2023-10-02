import json

from tqdm import tqdm
from PIL import Image, ImageDraw

import numpy as np


with open("infant-dataset/annotations.json", "r") as f:
    data = json.load(f)

sk_point = data["categories"][0]["skeleton"]

for idx, img_info in enumerate(data["images"]):
    img_path = "infant-dataset/" + img_info["file_name"]

    image = Image.open(img_path)

    keypoints = data["annotations"][idx]["keypoints"]
    keypoints = np.array(keypoints).reshape(-1, 3)

    draw = ImageDraw.Draw(image)

    for n1, n2 in tqdm(sk_point):
        x1, y1, _ = keypoints[n1]
        x2, y2, _ = keypoints[n2]
        draw.line((x1, y1, x2, y2), fill=(0, 0, 255), width=2)

    for i, kp in enumerate(keypoints):
        x, y, _ = kp
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 255), width=2)
        draw.text((x - 5, y - 5), str(i),)

    image.save(f"./skeleton/{idx}.jpg")

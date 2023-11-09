import json

from tqdm import tqdm
from PIL import Image, ImageDraw

import numpy as np


with open("infant-dataset/annotations.json", "r") as f:
    data = json.load(f)

sk_point = data["categories"][0]["skeleton"]
print(sk_point)

sk_point_map = {
    "head": [0, 1, 2, 3, 4, 5, 6, 7],
    "upper_body": [8, 9, 10, 11, 12, 13, 14, 15],
    "lower_body": [16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
}

for idx, img_info in enumerate(data["images"]):
    img_path = "infant-dataset/" + img_info["file_name"]

    image = Image.open(img_path)

    keypoints = data["annotations"][idx]["keypoints"]
    keypoints = np.array(keypoints).reshape(-1, 3)

    draw = ImageDraw.Draw(image)

    for n1, n2 in sk_point:
        x1, y1, v1 = keypoints[n1]
        x2, y2, v2 = keypoints[n2]
        if n1 in sk_point_map["head"]:
            draw.line((x1, y1, x2, y2), fill=(0, 0, 255), width=2)
        if n1 in sk_point_map["upper_body"]:
            draw.line((x1, y1, x2, y2), fill=(0, 255, 0), width=2)
        if n1 in sk_point_map["lower_body"]:
            draw.line((x1, y1, x2, y2), fill=(255, 0, 0), width=2)

    for i, kp in enumerate(keypoints):
        x, y, v = kp
        # v = 0  /  v = 1  /  v = 2
        if v == 2:  # 검은색
            if i in sk_point_map["head"]:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 0), width=2)
            if i in sk_point_map["upper_body"]:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 0), width=2)
            if i in sk_point_map["lower_body"]:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(0, 0, 0), width=2)
            draw.text(
                (x - 5, y - 5),
                str(i),
            )
        if v == 1:  # 주황색
            if i in sk_point_map["head"]:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 100, 0), width=2)
            if i in sk_point_map["upper_body"]:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 100, 0), width=2)
            if i in sk_point_map["lower_body"]:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 100, 0), width=2)
            draw.text(
                (x - 5, y - 5),
                str(i),
            )
        if v == 0:  # 보라색
            if i in sk_point_map["head"]:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 255), width=2)
            if i in sk_point_map["upper_body"]:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 255), width=2)
            if i in sk_point_map["lower_body"]:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 0, 255), width=2)
            draw.text(
                (x - 5, y - 5),
                str(i),
            )

    image.save(f"./color_skeleton/{idx}.jpg")

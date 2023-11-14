import json
import numpy as np
import os
import glob
import shutil
import re

from collections import defaultdict
from PIL import Image
from tqdm import tqdm

json_idx = [17962, 18027, 18028, 18030, 18222]

image_list = []
for j in json_idx:
    imgs = glob.glob(f"original-data/images/{j}/*/*.jpg")
    imgs = sorted(imgs, key=lambda x: int(re.sub(r"[^0-9]", "", x.split("_")[-1])))
    image_list += imgs

json_files = glob.glob("./original-data/*.json")
idx = 0

new_annot_file = {"images": [], "annotations": [], "categories": []}
new_annot_file["categories"].append(
    {
        "id": 1,
        "name": "infant",
        "supercategory": None,
        "keypoints": [
            "head_top",
            "right_eye",
            "left_eye",
            "nose",
            "mouth",
            "right_ear",
            "left_ear",
            "chin",
            "notch_of_sternum",
            "xiphoid_process",
            "right_shoulder_joint",
            "left_shoulder_joint",
            "right_elbow_joint",
            "left_elbow_joint",
            "right_wrist_joint",
            "left_wrist_joint",
            "mid hip",
            "right_pelvis, ASIS",
            "left_pelvis, ASIS",
            "public_symphysis",
            "right_hip_joint",
            "left_hip_joint",
            "right_knee_joint",
            "left_knee_joint",
            "right_ankle_joint",
            "left_ankle_joint",
        ],
        "skeleton": [
            (0, 3),
            (1, 3),
            (2, 3),
            (1, 5),
            (2, 6),
            (3, 4),
            (4, 7),
            (7, 8),
            (8, 9),
            (8, 10),
            (8, 11),
            (10, 12),
            (11, 13),
            (12, 14),
            (13, 15),
            (8, 16),
            (16, 17),
            (16, 18),
            (16, 19),
            (19, 20),
            (19, 21),
            (20, 22),
            (21, 23),
            (22, 24),
            (23, 25),
        ],
    },
)

for js, json_file in enumerate(json_files):
    with open(f"{json_files[js]}", "r") as f:
        annots = json.load(f)

    annot_dict = defaultdict(list)

    for i, annot in enumerate(annots["annotations"]):
        annot_dict[annot["id"]].append(annot)

        for k_idx, point in enumerate(annots["annotations"][i]["keypoints"]):
            k_idx += 1
            if k_idx % 3 == 2:
                if annots["annotations"][i]["keypoints"][k_idx] == 0:
                    annots["annotations"][i]["keypoints"][k_idx] = 1
                elif annots["annotations"][i]["keypoints"][k_idx] == 1:
                    annots["annotations"][i]["keypoints"][k_idx] = 2

    annot_id = 1
    for image_info in tqdm(annots["images"]):
        file_name = image_list[idx].replace(
            image_list[idx].split("_")[-1], f'{image_info["id"]}.jpg'
        )
        modified_path = file_name.replace("original-data/", "")
        modified_path = modified_path.split("\\")
        del modified_path[-2]
        modified_path = "/".join(modified_path)
        w, h = Image.open(file_name).convert("RGB").size
        image_result = {
            "id": idx + 1,
            "width": w,
            "height": h,
            "file_name": modified_path,
        }
        new_annot_file["images"].append(image_result)

        for annot in annot_dict[image_info["id"]]:
            kp = np.array(annot["keypoints"]).reshape(-1, 3)
            x1, y1, x2, y2 = (
                kp[:, 0].min(),
                kp[:, 1].min(),
                kp[:, 0].max(),
                kp[:, 1].max(),
            )
            w, h = x2 - x1, y2 - y1
            annot_result = {
                "id": idx + 1,
                "image_id": idx + 1,
                "category_id": 1,
                "bbox": [x1, y1, w, h],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
                "segmentations": [],
                "keypoints": annot["keypoints"],
                "num_keypoints": int(kp.shape[0]),
            }
            new_annot_file["annotations"].append(annot_result)
        annot_id += 1
        idx += 1


with open("infant-dataset/annotations.json", "w", encoding="utf-8") as f:
    json.dump(new_annot_file, f, indent="\t")

img_path = [
    "./original-data/images/17962/126524578/",
    "./original-data/images/18027/126499894/",
    "./original-data/images/18028/126500381/",
    "./original-data/images/18030/126507318/",
    "./original-data/images/18222/126720846/",
]

for i in range(5):
    jpgs = os.listdir(img_path[i])

    new_path = f"./infant-dataset/images/{json_idx[i]}/"
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for jpg in tqdm(jpgs):
        shutil.copy(img_path[i] + jpg, new_path + jpg)

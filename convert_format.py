import json
import numpy as np
import os
import glob
import shutil
import sys
import re

from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# !TODO
# 1. infant-dataset/ annotations.json 파일 하나를 생성한다.
# 1.1 original-data에 있는 숫자.json 파일들을 하나로 합쳐야 된다.
# 1.1 id가 중복되어서 중복안되게끔 생성해줘야하고, 파일 경로 상대로 다시 셋팅해줘되고,
# 그다음에 그 경로대로 original/ 폴더에 있는 이미지 infant-dataset 경로 복사해주기
# 2. 키 포인트 추출
# 2.1 annotations > keypoints이용해서 x,y 좌표얻음
# 2.2 그 좌표를 통해 4개 좌표 뽑고 박스치기 (# x, y, w, h)


"""
{
    "images": [
        {
            "id": 1,
            "width": 1280,
            "height": 720,
            "file_name": "100/001/aa_1_1.jpg"
        },
        {},...
    ],
    "annotations: [
        {
            
        },
        {},...
    ],
    "categories": [
        
    ]
}

 {
      "id": 166, # 이거는 이미지 id랑 매칭함 -> "image_id"
      "category_id": 1,
      "keypoints": [
        559.5286757490586,
        122.93845280185698,
        0,
        549.736017007769,
        179.511974103473,
        0,...
      ],
      "num_keypoints": "int"
    },

{
    "id": 121431243,
    "image_id": 1, 
    "category_id": 1, # 사람인, 배인지, 차인지? 사람만 있기 1
    "bbox": [[x, y, w, h]],
    "area": float,
    "iscrowd": 0,
    "segmentations": None,
    "keypoints": [
        559.5286757490586,
        122.93845280185698,
        0,
        549.736017007769,
        179.511974103473,
        0,...
    ],
    "num_keypoints": 25
},
"""

'''
original-data/images/17962/126524578/126524578_1.jpg
original-data/images/18027/126499894/126499894_1.jpg
original-data/images/18028/126500381/126500381_1.jpg
original-data/images/18030/126507318/126507318_1.jpg
original-data/images/18222/126720846/126720846_1.jpg
'''

json_idx = [17962, 18027, 18028, 18030, 18222]

image_list = []
for j in json_idx:
    imgs = glob.glob(f'original-data/images/{j}/*/*.jpg')
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
        "skeleton": [],
    },
)

for js, json_file in enumerate(json_files):
    print('json: ', json_file)
    with open(f"{json_files[js]}", "r") as f:
        annots = json.load(f)

    annot_dict = defaultdict(list)

    for annot in annots["annotations"]:
        annot_dict[annot["id"]].append(annot)
        

    annot_id = 1
    for image_info in (annots["images"]):
        print(idx)
        file_name = image_list[idx].replace(image_list[idx].split('_')[-1], f'{image_info["id"]}.jpg')
        modified_path = file_name.split("\\")
        del modified_path[-2]
        # print(modified_path)
        modified_path = "/".join(modified_path)
        # modified_path = os.path.join(modified_path[0], modified_path[1])
        print(modified_path)
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
            x1, y1, x2, y2 = kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()
            w, h = x2 - x1, y2 - y1
            annot_result = {
                "id": idx + 1,
                "image_id": idx + 1,
                "category_id": 1,
                "bbox": [[x1, y1, w, h]],
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
    "./original-data/images/18222/126720846/"
]

for i in range(5):
    jpgs = os.listdir(img_path[i])

    new_path = f"./infant-dataset/images/{json_idx[i]}/"
    if not os.path.exists(new_path):
        os.mkdir(new_path)

    for jpg in jpgs:
        print(img_path[i] + jpg)
        print(new_path + jpg)
        shutil.copy(img_path[i] + jpg, new_path + jpg)

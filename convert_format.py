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

with open("original-data/17962.json", "r") as f:
    annots = json.load(f)

annot_dict = defaultdict(list)

for annot in annots["annotations"]:
    annot_dict[annot["id"]].append(annot)

new_annot_file = {"images": [], "annotations": [], "categories": []}
new_annot_file["categories"].append(
    {
        "id": 1, "name": "infant", "supercategory": None,
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
            "left_ankle_joint"
        ],
        "skeleton": [
        ]
    },
)


annot_id = 1
for image_info in tqdm(annots["images"]):
    file_name = f"original-data/images/17962/126524578/126524578_{image_info['id']}.jpg" 
    modified_path = file_name.split('/')
    del modified_path[-2]
    modified_path = '/'.join(modified_path)
    w, h = Image.open(file_name).convert("RGB").size
    image_result = {
        "id": image_info["id"],
        "width": w,
        "height": h,
        "file_name": modified_path,
    }
    new_annot_file["images"].append(image_result)
    for annot in annot_dict[image_info["id"]]:
        kp = np.array(annot["keypoints"]).reshape(-1, 3)
        x1, y1, x2, y2 = kp[:, 0].min(), kp[:, 1].min(), kp[:, 0].max(), kp[:, 1].max()
        annot_result = {
            "id": annot_id,
            "image_id": image_info["id"],
            "category_id": 1,
            "bbox": [[x1, y1, x2, y2]],  
            "area": (x2 - x1) * (y2 - y1),
            "iscrowd": 0,
            "segmentations": [],
            "keypoints": annot["keypoints"],
            "num_keypoints": int(kp.shape[0]),
        }
        new_annot_file["annotations"].append(annot_result)
    annot_id += 1
    
with open("infant-dataset/annotations.json", "w", encoding="utf-8") as f:
    json.dump(new_annot_file, f, indent="\t")

# paths = glob.glob("./original-data/*/*/*/*")
# #

# # img_names = []

# # for i in range(1,6):
# #     for val in path_data[f"path_data{i}"]:
# #         print(val)

# # 상단경로 얻기
# path_names = []
# for i in range(1, 1000, 200):
#     p1, p2 = Path(paths[i]).parts[-3:-1]
#     path_names.append(p1 + "/" + p2)
# # >>> ['17962/126524578', '18027/126499894', '18028/126500381', '18030/126507318', '18222/126720846']

# # path_1, path_2 = Path(paths[0]).parts[-3:-1]
# # path_17962 = path_1 + '/' + path_2


# # jpg만 얻기
# img_names = []
# for i in range(0, 999, 200):
#     imgs = []
#     for j in range(i, i + 200):
#         imgs.append(Path(paths[j]).parts[-1])
#     imgs = sorted(imgs, key=lambda x: int(re.sub(r"[^0-9]", "", x.split("_")[-1])))
#     img_names += imgs
# # >>> 5 x 200 ea

# # img_17962 = []
# # for i in range(200):
# #     img_17962.append(Path(paths[i]).parts[-1])
# # img_17962 = sorted(img_17962, key=lambda x: int(re.sub(r"[^0-9]", "", x.split("_")[-1])))

# # name_list = sorted(
# #     qwer, key=lambda x: int(re.sub(r"[^0-9]", "", x.split("_")[-1]))
# # )


# # 최종경로(file_name) 얻기
# file_names = []
# for idx, i in enumerate(range(0, 999, 200)):
#     for name in img_names[i : i + 200]:
#         file_names.append(path_names[idx] + "/" + name)
# # print(file_names)


# print(Path(paths[0]).parts[-3])
# # json 파일 이름
# json_file_names = []
# for i in range(5):
#     json_file_name = f"{Path(paths[i]).parts[-3]}.json"
#     json_file_names.append(json_file_name)
# print(json_file_names)

# sys.exit()

# annot_by_ids = defaultdict(list)
# annot_by_ids[1].append({"images": {}, "annotations": {}})
# annot_by_ids[2].append({"what": "the"})
# ids = 1

# path_list = path.split("/")

# print(len(paths))
# path_data = {'path_data1': paths[:200],
#              'path_data2': paths[200:400],
#              'path_data3': paths[400:600],
#              'path_data4': paths[600:800],
#              'path_data5': paths[800:]}
# path_data["path_data1"]

# new_name_list = []
# for i in range(len(name_list)):
#     img_dir = path_list[-2]
#     a = name_list[i].split("_")[-1]
#     img_name = img_dir + "_" + a

#     new_name_list.append(img_name)


# # 뽑은 파일 이름으로 json 덮어쓰기
# with open("infant-dataset/images/17962.json") as f:
#     data = json.loads(f.read())

# for i in range(len(data["images"])):
#     data["images"][i]["file_name"] = new_name_list[i]


# print(data["images"][0]["file_name"])
# print(data["images"][10]["file_name"])

# with open("infant-dataset/images/17962.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, indent="\t")

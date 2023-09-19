import json
import numpy as np
import os
import glob
import shutil
import sys
from pathlib import Path
from collections import defaultdict

# !TODO 
# 1. infant-dataset/ annotations.json 파일 하나를 생성한다.
# 1.1 original-data에 있는 숫자.json 파일들을 하나로 합쳐야 된다.
# 1.1 id가 중복되어서 중복안되게끔 생성해줘야하고, 파일 경로 상대로 다시 셋팅해줘되고, 
# 그다음에 그 경로대로 original/ 폴더에 있는 이미지 infant-dataset 경로 복사해주기
# 2. 키 포인트 추출
# 2.1 annotations > keypoints이용해서 x,y 좌표얻음
# 2.2 그 좌표를 통해 4개 좌표 뽑고 박스치기


# # json 수정할 파일 이름 뽑아보기
# with open("infant-dataset/images/17962.json") as f:
#     data = json.loads(f.read())
# # print(type(data["images"][0]))
# for i in range(len(data["images"])):
#     file_name = data["images"][i]["file_name"]  # json에서 파일이름을 뽑은거
#     print(file_name)


import re
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

path = "./infant-dataset/images/PROJ-6312_keypoint_2023-08-04/17962/126524578/*.jpg"

paths = glob.glob("./original-data/*/*/*")
print(f"{Path(paths[0]).parts[-2]}.json")

annot_by_ids = defaultdict(list)
annot_by_ids[1].append({
    "images": {},
    "annotations": {}
})
annot_by_ids[2].append({"what": "the"})
print(annot_by_ids)
print(annot_by_ids[1])
with open("infant-dataset/annotations.json", "w") as f:
    json.dumps(f)

sys.exit()

ids = 1

path_list = path.split("/")

name_list = sorted(
    glob.glob(path), key=lambda x: int(re.sub(r"[^0-9]", "", x.split("_")[-1]))
)

new_name_list = []
for i in range(len(name_list)):
    img_dir = path_list[-2]
    a = name_list[i].split("_")[-1]
    img_name = img_dir + "_" + a
    # print(img_name)
    new_name_list.append(img_name)
# print(new_name_list)

# 뽑은 파일 이름으로 json 덮어쓰기
with open("infant-dataset/images/17962.json") as f:
    data = json.loads(f.read())

for i in range(len(data["images"])):
    data["images"][i]["file_name"] = new_name_list[i]


print(data["images"][0]["file_name"])
print(data["images"][10]["file_name"])

with open("infant-dataset/images/17962.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent="\t")



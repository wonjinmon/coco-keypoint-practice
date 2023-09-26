import os
import json
import shutil as sh
from glob import glob
from PIL import Image


def check_dir_exist(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(os.path.join(path, "images")):
        os.makedirs(os.path.join(path, "images"))
    if not os.path.exists(os.path.join(path, "labels")):
        os.makedirs(os.path.join(path, "labels"))


def main(dst_path: str, root_path: str):
    # Make directory
    check_dir_exist("dataset")

    # Define convert to coco format
    convert_to_coco = {}
    categories = [ 
        {
            "id": 1,
            "name": "infant",
            "supercategory": "person",
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
                (0, 3), (1, 3), (2, 3), (1, 5), (2, 6), (3, 4), (4, 7), 
                (7, 8), (8, 9), (8, 10), (8, 11), (10, 12), (11, 13), (12, 14), 
                (13, 15), (8, 16), (16, 17), (16, 18), (16, 19), (19, 20), 
                (19, 21), (20, 22), (21, 23), (22, 24), (23, 25)
            ]
        },
    ]
    images = []
    annotations = []
    annot_idx = 1

    # loop each json file
    json_paths = glob(os.path.join(root_path, "*.json"))
    for json_path in json_paths:
        with open(json_path, "r") as f:
            raw_json = json.load(f)

        # Get raw_data json file
        folder_num = json_path.split("\\")[-1].split(".")[0]
        raw_images = raw_json["images"]
        raw_annotations = raw_json["annotations"]

        samples = glob(os.path.join(root_path, str(folder_num), "*", "*.jpg"))
        samples = sorted(samples, key=lambda x: int(x.split("\\")[-1].split(".")[0].split('_')[1]))
        image_name = samples[0].split("\\")[-2]

        for i, raw_image in enumerate(raw_images):
            # Extract width, height with PIL 
            # image = Image.open(samples[i])
            # width, height = image.size

            # Extract image info and copy image to dst path
            img_obj = {}
            image_id = raw_image["id"]
            width, height = raw_image["width"], raw_image["height"]
            file_name = os.path.join(dst_path, "images", image_name, f"{image_id}.jpg")
            file_name = file_name.replace("\\", "/")
            
            img_obj["id"] = annot_idx
            img_obj["width"] = width
            img_obj["height"] = height
            img_obj["file_name"] = file_name
            images.append(img_obj)

            # make dst directory
            if not os.path.exists(os.path.join(dst_path, "images", image_name)):
                os.makedirs(os.path.join(dst_path, "images", image_name))

            # copy to dst path
            sh.copy(samples[i], file_name)

            # Extract annotation info
            raw_annots = list(filter(lambda x: x["id"]==image_id, raw_annotations))

            for raw_annot in raw_annots:
                annot_obj = {}
                annot_obj["id"] = annot_idx
                annot_obj["image_id"] = annot_idx
                annot_obj["category_id"] = 1
                annot_obj["keypoints"] = raw_annot["keypoints"]
                annot_obj["num_keypoints"] = 26
                annotations.append(annot_obj)
                annot_idx += 1

    convert_to_coco["categories"] = categories
    convert_to_coco["images"] = images
    convert_to_coco["annotations"] = annotations

    with open(os.path.join(dst_path, "test.json"), "w") as f:
        json.dump(convert_to_coco, f)


if __name__ == '__main__':
    main("dataset", "raw_dataset")
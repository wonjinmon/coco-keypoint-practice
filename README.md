# convert coco keypoint format

# Should be a format

```json
{
    "image": [
        {
            "id": "int",
            "width": "int",
            "height": "int",
            "file_name": "str"
        }
    ],
    "annotation": [
        {
            "id": "int",
            "image_id": "int",
            "category_id": "int",
            "segmentation": "RLE or [polygon]", "area": "float", 
            "bbox": "[x,y,width,height]", "iscrowd": "0 or 1",
            "keypoints": "[x1, y1, v1,...]",
            "num_keypoints": "int"
        }
    ],
    "categories" : [
        {
            "id": "int",
            "name": "str",
            "supercategory": "str",
            "keypoints": "[str]",
            "skeleton": "[edge]"
        }
    ]
}
```
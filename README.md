# convert coco keypoint format

# Should be a format

```
infant-dataset/
    images/
        17962/
            126524578_1.jpg
            ...
        18027/
            126499894_1.jpg
            ...
        18028/
        ...
        18030/
        ...
        18222/
        ...
    annotations.json
```


```json
{
    "images": [
        {
            "id": "int",
            "width": "int",
            "height": "int",
            "file_name": "images/17962/126524578_1.jpg",
            "license": "int", 
            "flickr_url": "str", 
            "coco_url": "str", 
            "date_captured": "datetime",
        }
    ],
    "annotations": [
        {
            "id": "int",
            "image_id": "int",
            "category_id": "int",
            "segmentation": "RLE or [polygon]", 
            "area": "float", 
            "bbox": "[x,y,width,height]", 
            "iscrowd": "0 or 1",
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
### Visualization

"draw_bbox" 

"draw_keypoints" 

"draw_skeleton" 

"draw_color_skeleton"
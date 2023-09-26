import json

from tqdm import tqdm
from PIL import Image, ImageDraw


with open("infant-dataset/annotations.json", "r") as f:
    data = json.load(f)

# print(len(data["images"]))

# random.choice()

for idx, img_info in tqdm(enumerate(data["images"])):
    img_path = "infant-dataset/" + img_info["file_name"]
    x, y, w, h = data["annotations"][idx]["bbox"][0]
    x1, y1, x2, y2 = x, y, x + w, y + h
    image = Image.open(img_path)

    draw = ImageDraw.Draw(image)
    draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=3)
    # p = img_info["file_name"].split('.')[0]
    # save_path = './save/' + p
    image.save(f'./bbox/{str(img_info["id"]) + "_" + str(1 + idx)}.jpg')


# x, y, w, h = data["annotations"][0]["bbox"][0]
# x1, y1, x2, y2 = x, y, x+w, y+h

# img = Image.open(img_path)
# # img.show()

# draw = ImageDraw.Draw(img)
# draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 0), width=2)
# # img.show()

# ./save/images/17962/126524578_1.jpg

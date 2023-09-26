import json

from tqdm import tqdm
from PIL import Image, ImageDraw


with open("infant-dataset/annotations.json", "r") as f:
    data = json.load(f)


print(data["categories"][0]["skeleton"])

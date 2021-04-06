# %%
import os
import math
from PIL import Image, ImageDraw

sequences = os.listdir("sequences")
print("Found sequences:")
# %%
def idx(str) -> int:
    return int(str.lstrip("0"))

def calc_bbox(center,side_length):
    length_split = int(math.ceil(side_length/2))
    upleft = (center[0] - length_split, center[1] - length_split)
    lowright =(center[0] + length_split, center[1] + length_split)
    return [*upleft, *lowright]

bbox_sizes = {}
with open("bbox_size.csv",'r') as bbfile:
    lines = bbfile.readlines()
    for line in lines:
        sequence_idx = int(idx(line.split(",")[0]))
        side_width = int(line.split(",")[1])
        bbox_sizes[sequence_idx] = side_width

for sequence in sequences:
    images = os.listdir(os.path.join("sequences",sequence))
    images = [images[0]]
    annotation_path = os.path.join("annotations", f"{sequence}.txt")
    if not os.path.isfile(annotation_path):
        print(f"WARINING: Sequence not found: {annotation_path}")
        continue
    annotation_file = open(os.path.join("annotations", f"{sequence}.txt"),"r")
    annotations = {}
    for row in annotation_file.readlines():
        row_split = row.split(",")
        image_idx = int(row_split[0].lstrip("0"))
        width = int(row_split[1])
        height = int(row_split[2])
        points = annotations.get(image_idx)
        if points is None:
            points = []
        points.append((width, height))
        annotations[image_idx] = points
    
    for image_name in images:
        print(f"Loading image {sequence}/{image_name}")
        img =  Image.open(os.path.join("sequences",sequence, image_name))
        draw = ImageDraw.Draw(img)
        heads = annotations[int(image_name.lstrip("0").replace(".jpg",""))]
        draw.point(heads,'red')
        bboxes = list(map(lambda center: calc_bbox(center, bbox_sizes[idx(sequence)]), heads))
        for bbox in bboxes:
            draw.rectangle(bbox,outline="red")
        img.show()
        input_char = input("Press q to quit, any other button will load next image:")
        img.close()
        if input_char == "q":
            exit()
# %%

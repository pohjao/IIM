import os
import math
import json
import shutil
import random
import numpy as np
from typing import List

### Utility functions ###
def idx(str) -> int:
    return int(str.lstrip("0"))

def calc_bbox(center,side_length):
    length_split = int(math.ceil(side_length/2))
    upleft = (center[0] - length_split, center[1] - length_split)
    lowright =(center[0] + length_split, center[1] + length_split)
    return [*upleft, *lowright]

def create_gt_loc(meta) -> List[str]:
    values = []
    values.append(meta["img_id"].replace(".jpg",""))
    values.append(str(meta["human_num"]))
    for head in meta["boxes"]:
        x1, y1, x2, y2 = int(head[0]), int(head[1]), int(head[2]), int(head[3])
        center_x, center_y, w, h = int((x1+x2)/2), int((y1+y2)/2),  int((x2-x1)),int((y2-y1)),
        area = w * h
        if area == 0:
            continue

        level_area = 0
        if area >= 1 and area < 10:
            level_area = 0
        elif area > 10 and area < 100:
            level_area = 1
        elif area > 100 and area < 1000:
            level_area = 2
        elif area > 1000 and area < 10000:
            level_area = 3
        elif area > 10000 and area < 100000:
            level_area = 4
        elif area > 100000:
            level_area = 5

        r_small = int(min(w, h) / 2)
        r_large = int(np.sqrt (w * w + h * h) / 2)


        values.append(str(center_x))
        values.append(str(center_y))
        values.append(str(r_small))
        values.append(str(r_large))
        values.append(str(level_area))
    return values

visdrone_export_path = "../ProcessedData/VisDrone"
images_export_path = os.path.join(visdrone_export_path, "images")
jsons_export_path = os.path.join(visdrone_export_path, "jsons")
gt_file_path = os.path.join(visdrone_export_path, "val_gt_loc.txt")
train_list_path = os.path.join(visdrone_export_path, "train.txt")
val_list_path = os.path.join(visdrone_export_path, "val.txt")
val_portion = 0.15

### Directory and file preparations ###
if not os.path.isdir(visdrone_export_path):
    os.mkdir(visdrone_export_path)

if not os.path.isdir(images_export_path):
    os.mkdir(images_export_path)

if not os.path.isdir(jsons_export_path):
    os.mkdir(jsons_export_path)

if os.path.isfile(gt_file_path):
    os.remove(gt_file_path)

if os.path.isfile(train_list_path):
    os.remove(train_list_path)

if os.path.isfile(val_list_path):
    os.remove(val_list_path)


### Read sequence bounding box sizes ###
bbox_sizes = {}
with open("bbox_size.csv",'r') as bbfile:
    lines = bbfile.readlines()
    for line in lines:
        sequence_idx = int(idx(line.split(",")[0]))
        side_width = int(line.split(",")[1])
        bbox_sizes[sequence_idx] = side_width

### Handle sequences ###
sequences = os.listdir("sequences")
for sequence in sequences:
    images = os.listdir(os.path.join("sequences",sequence))
    annotation_path = os.path.join("annotations", f"{sequence}.txt")
    if not os.path.isfile(annotation_path):
        print(f"WARINING: Sequence not found: {annotation_path}")
        continue
    annotation_file = open(annotation_path,"r")
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
    
    val_images = random.choices(images, k=int(len(images)*val_portion))
    for image_name in images:
        meta = {}
        exported_image_name = f"{sequence}_{image_name}"
        meta["img_id"] = exported_image_name
        heads = annotations[int(image_name.lstrip("0").replace(".jpg",""))]
        bboxes = list(map(lambda center: calc_bbox(center, bbox_sizes[idx(sequence)]), heads))
        meta["human_num"] = len(heads)
        meta["points"] = heads
        meta["boxes"] = bboxes

        export_image_src = os.path.join("sequences",sequence,image_name)
        export_image_dst = os.path.join(images_export_path,exported_image_name)

        shutil.copyfile(export_image_src, export_image_dst)
        with open(os.path.join(jsons_export_path, exported_image_name.replace(".jpg",".json")), 'w') as json_file:
            json.dump(meta, json_file)

        if image_name in val_images:
            with open(gt_file_path, "a+") as gt_file:
                row = " ".join(create_gt_loc(meta))
                row += "\n"
                gt_file.write(row)
            with open(val_list_path, "a+") as val_list_file:
                val_list_file.write(exported_image_name.replace(".jpg","")+"\n")
        else:
            with open(train_list_path, "a+") as train_list_file:
                train_list_file.write(exported_image_name.replace(".jpg","")+"\n")
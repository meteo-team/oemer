import os
import random
from PIL import Image

import cv2
import numpy as np

from .constant_min import CLASS_CHANNEL_MAP


HALF_WHOLE_NOTE = [39, 41, 42, 43, 45, 46, 47, 49]


def fill_hole(gt, tar_color):
    assert tar_color in HALF_WHOLE_NOTE
    tar = np.where(gt==tar_color, 1, 0).astype(np.uint8)
    cnts, _ = cv2.findContours(tar, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)

        # Scan by row
        for yi in range(y, y+h):
            cur = x
            cand_y = []
            cand_x = []
            while cur <= x+w:
                if tar[yi, cur] > 0:
                    break
                cur += 1
            while cur <= x+w:
                if tar[yi, cur] == 0:
                    break
                cur += 1
            while cur <= x+w:
                if tar[yi, cur] > 0:
                    break
                cand_y.append(yi)
                cand_x.append(cur)
                cur += 1
            if cur <= x+w:
                cand_y = np.array(cand_y)
                cand_x = np.array(cand_x)
                tar[cand_y, cand_x] = 1

        # Scan by column
        for xi in range(x, x+w):
            cur = y
            cand_y = []
            cand_x = []
            while cur <= y+h:
                if tar[cur, xi] > 0:
                    break
                cur += 1
            while cur <= y+h:
                if tar[cur, xi] == 0:
                    break
                cur += 1
            while cur <= y+h:
                if tar[cur, xi] > 0:
                    break
                cand_y.append(cur)
                cand_x.append(xi)
                cur += 1
            if cur <= y+h:
                cand_y = np.array(cand_y)
                cand_x = np.array(cand_x)
                tar[cand_y, cand_x] = 1

    return tar


def build_label(seg_path):
    img = Image.open(seg_path)
    arr = np.array(img)
    color_set = set(np.unique(arr))
    color_set.remove(0)  # Remove background color from the candidates

    total_chs = len(set(CLASS_CHANNEL_MAP.values())) + 2  # Plus 'background' and 'others' channel.
    output = np.zeros(arr.shape + (total_chs,))

    output[..., 0] = np.where(arr==0, 1, 0)
    for color in color_set:
        ch = CLASS_CHANNEL_MAP.get(color, -1)
        if (ch != 0) and color in HALF_WHOLE_NOTE:
            note = fill_hole(arr, color)
            output[..., ch] += note
        else:
            output[..., ch] += np.where(arr==color, 1, 0)
    return output


def find_example(color: int, max_count=100, mark_value=200):
    dataset_path = "/media/kohara/ADATA HV620S/dataset/ds2_dense/segmentation"
    files = os.listdir(dataset_path)
    random.shuffle(files)
    for ff in files[:max_count]:
        path = os.path.join(dataset_path, ff)
        img = Image.open(path)
        arr = np.array(img)
        if color in arr:
            return np.where(arr==color, mark_value, arr)


if __name__ == "__main__":
    seg_folder = '/media/kohara/ADATA HV620S/dataset/ds2_dense/segmentation'
    files = os.listdir(seg_folder)
    path = os.path.join(seg_folder, random.choice(files))
    #out = build_label(path)

    color = 45
    arr = find_example(color)
    arr = np.where(arr==200, color, arr)
    out = fill_hole(arr, color)



import cv2
import numpy as np

from . import layers
from .bbox import *
from .utils import get_unit_size
from .general_filtering_rules import filter_out_of_range_bbox


def get_degree(line):
    return np.rad2deg(np.arctan2(line[3] - line[1], line[2] - line[0]))


def filter_lines(lines, min_degree=75):
    staffs = layers.get_layer('staffs')

    lines = filter_out_of_range_bbox(lines)
    min_y = min([st.y_upper for st in staffs.reshape(-1, 1).squeeze()])
    max_y = max([st.y_lower for st in staffs.reshape(-1, 1).squeeze()])

    cands = []
    for line in lines:
        degree = get_degree(line)
        if degree < min_degree:
            continue

        if line[1] < min_y or line[3] > max_y:
            continue

        cands.append(line)
    return cands


def get_barline_map(symbols, bboxes):
    img = np.zeros_like(symbols)
    for box in bboxes:
        box = list(box)
        if box[2]-box[0] == 0:
            box[2] += 1
        img[box[1]:box[3], box[0]:box[2]] += symbols[box[1]:box[3], box[0]:box[2]]
    img[img>1] = 1
    return img


def get_barline_box(bmap):
    ker = np.ones((5, 2), dtype=np.uint8)
    ext_bmap = cv2.erode(cv2.dilate(bmap.astype(np.uint8), ker), ker)
    bboxes = get_bbox(ext_bmap)

    valid_box = []
    heights = []
    for box in bboxes:
        unit_size = get_unit_size(*get_center(box))
        h = box[3] - box[1]
        if h > unit_size:
            heights.append(h)
            valid_box.append(box)

    return valid_box


def draw_lls(lines, ori_img):
    img = to_rgb_img(ori_img)
    for line in lines:
        degree = get_degree(line)
        cv2.rectangle(img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
        msg = f"{degree:.2f}"
        cv2.putText(img, msg, (line[2]+2, line[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    return img


if __name__ == "__main__":
    symbols = layers.get_layer('symbols_pred')
    stems = layers.get_layer('stems_rests_pred')
    notehead = layers.get_layer('notehead_pred')
    clefs = layers.get_layer('clefs_keys_pred')
    staffs = layers.get_layer('staffs')

    mix = symbols - stems - notehead - clefs
    mix[mix<0] = 0

    lines = find_lines(mix)
    lines = filter_lines(lines)
    bmap = get_barline_map(symbols, lines) + stems
    bmap[bmap>1] = 1
    bboxes = get_barline_box(bmap)

    bmap = to_rgb_img(bmap)
    for box in bboxes:
        cv2.rectangle(bmap, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        unit_size = get_unit_size(*get_center(box))
        ratio = (box[3] - box[1]) / unit_size
        if ratio > 9:
            cv2.putText(bmap, f"{ratio:.2f}", (box[2]+2, box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

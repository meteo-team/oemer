import os
import logging

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

from . import layers


def get_logger(name, level="warn"):
    """Get the logger for printing informations.
    Used for layout the information of various stages while executing the program.
    Set the environment variable ``LOG_LEVEL`` to change the default level.
    Parameters
    ----------
    name: str
        Name of the logger.
    level: {'debug', 'info', 'warn', 'warning', 'error', 'critical'}
        Level of the logger. The level 'warn' and 'warning' are different. The former
        is the default level and the actual level is set to logging.INFO, and for
        'warning' which will be set to true logging.WARN level. The purpose behind this
        design is to categorize the message layout into several different formats.
    """
    logger = logging.getLogger(name)
    level = os.environ.get("LOG_LEVEL", level)

    msg_formats = {
        "debug": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
        "info": "%(asctime)s %(message)s  [at %(filename)s:%(lineno)d]",
        "warn": "%(asctime)s %(message)s",
        "warning": "%(asctime)s %(message)s",
        "error": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
        "critical": "%(asctime)s [%(levelname)s] %(message)s  [at %(filename)s:%(lineno)d]",
    }
    level_mapping = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    date_format = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=msg_formats[level.lower()], datefmt=date_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    if len(logger.handlers) > 0:
        rm_idx = [idx for idx, handler in enumerate(logger.handlers) if isinstance(handler, logging.StreamHandler)]
        for idx in rm_idx:
            del logger.handlers[idx]
    logger.addHandler(handler)
    logger.setLevel(level_mapping[level.lower()])
    return logger


def count(data, intervals):
    """Count elements in different intervals"""
    occur = []
    data = np.sort(data)
    intervals = np.insert(intervals, [0, len(intervals)], [np.min(data), np.max(data)])
    for idx in range(len(intervals[:-1])):
        sub = data[data>=intervals[idx]]
        sub = sub[sub<intervals[idx+1]]
        occur.append(len(sub))
    return occur


def find_closest_staffs(x, y):  # -> Tuple([Staff, Staff]):
    staffs = layers.get_layer('staffs')

    staffs = staffs.reshape(-1, 1).squeeze()
    diffs = sorted(staffs, key=lambda st: st - [x, y])
    if len(diffs) == 1:
        return diffs[0], diffs[0]
    elif len(diffs) == 2:
        return list(diffs)

    # There are over three candidates
    first = diffs[0]
    second = diffs[1]
    third = diffs[2]
    if abs(first.y_lower - y) <= abs(first.y_upper - y):
        # Closer to the lower bound of the first candidate.
        if second.y_center > first.y_center:
            return first, second
        elif third.y_center > first.y_center:
            return first, third
        else:
            return first, first
    else:
        # Closer to the upper bound of the first candidate.
        if second.y_center < first.y_center:
            return first, second
        elif third.y_center < first.y_center:
            return first, third
        else:
            return first, first


def get_unit_size(x, y):
    st1, st2 = find_closest_staffs(x, y)
    if st1.y_center == st2.y_center:
        return st1.unit_size

    # Within the stafflines
    if st1.y_upper <= y <= st1.y_lower:
        return st1.unit_size 

    # Outside stafflines.
    # Infer the unit size by linear interpolation.
    dist1 = abs(y - st1.y_center)
    dist2 = abs(y - st2.y_center)
    w1 = dist1 / (dist1 + dist2)
    w2 = dist2 / (dist1 + dist2)
    unit_size = w1 * st1.unit_size + w2 * st2.unit_size
    return unit_size


def get_global_unit_size():
    staffs = layers.get_layer('staffs')
    usize = []
    for st in staffs.reshape(-1, 1).squeeze():
        usize.append(st.unit_size)
    layers._global_unit_size = sum(usize) / len(usize)
    return layers._global_unit_size


def get_total_track_nums():
    staffs = layers.get_layer('staffs')
    tracks = [st.track for st in staffs.reshape(-1, 1).squeeze()]
    return len(set(tracks))


def remove_stems(data):
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    return cv2.dilate(cv2.erode(data.astype(np.uint8), ker), ker)


def estimate_degree(points, **kwargs):
    """Accepts list of (x, y) coordinates."""
    points = np.array(points)
    model = LinearRegression(**kwargs)
    model.fit(points[:, 0].reshape(-1, 1), points[:, 1])
    return slope_to_degree(model.coef_[0])


def slope_to_degree(y_diff, x_diff):
    return np.rad2deg(np.arctan2(y_diff, x_diff))

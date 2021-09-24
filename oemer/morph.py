
import numpy as np
import cv2


def get_kernel(kernel):
    if isinstance(kernel, tuple):
        # It's kernel shape
        kernel = np.ones(kernel, dtype=np.uint8)
    return kernel


def morph_open(img, kernel):
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_OPEN, ker)


def morph_close(img, kernel):
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_CLOSE, ker)


def morph_hit_miss(img, kernel):
    ker = get_kernel(kernel)
    return cv2.morphologyEx(img.astype(np.uint8), cv2.MORPH_HITMISS, ker)


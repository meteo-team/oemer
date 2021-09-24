CLASS_CHANNEL_LIST = [
    [165, 2],  # staff, ledgerLine
    [35, 37, 38, 39, 41, 42, 43, 45, 46, 47, 49, 52],  # notehead, stem
    [
        64, 58, 60, 66, 63, 69, 68, 61, 62, 67, 65, 59, 146,  # flags, beam
        97, 100, 99, 98, 101, 102, 103, 104, 96, 163,  # rests
        80, 78, 79, 74, 70, 72, 76, 3,  # sharp, flat, natural, barline
        10, 13, 12, 19, 11, 20, 51, # clefs, augmentationDot, 
        25, 24, 29, 22, 23, 28, 27, 34, 30, 21, 33, 26,  # timeSigs
    ]
]

CLASS_CHANNEL_MAP = {
    color: idx+1
    for idx, colors in enumerate(CLASS_CHANNEL_LIST)
    for color in colors
}

CHANNEL_NUM = len(CLASS_CHANNEL_LIST) + 2

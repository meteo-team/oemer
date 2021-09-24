from enum import Enum, auto


CLASS_CHANNEL_LIST = [
    [165, 2],  # staff, ledgerLine
    [35, 37, 38],  # noteheadBlack
    [39, 41, 42],  # noteheadHalf
    [43, 45, 46, 47, 49],  # noteheadWhole
    [64, 58, 59, 60, 66, 63, 69, 68, 61, 62, 67, 65],  # flags
    [146, 51],  # beam, augmentationDot
    [3, 52],  # barline, stem
    [74, 70, 72, 76],  # accidentalSharp, accidentalFlat, accidentalNatural, accidentalDoubleSharp
    [80, 78, 79],  # keySharp, keyFlat, keyNatural
    [97, 100, 99, 98, 101, 102, 103, 104, 96, 163],  # rests
    [136, 156, 137, 155, 152, 151, 153, 154, 149, 155],  # tuplets
    [145, 147],  # slur, tie
    [10, 13, 12, 19, 11, 20],  # clefs
    [25, 24, 29, 22, 23, 28, 27, 34, 30, 21, 33, 26],  # timeSigs
]

CLASS_CHANNEL_MAP = {
    color: idx+1
    for idx, colors in enumerate(CLASS_CHANNEL_LIST)
    for color in colors
}

CHANNEL_NUM = len(CLASS_CHANNEL_LIST) + 2


class NoteHeadConstant:
    NOTEHEAD_MORPH_WIDTH_FACTOR = 0.5 #0.444444  # Width to unit size factor when morphing
    NOTEHEAD_MORPH_HEIGHT_FACTOR = 0.4 #0.37037  # Height to unit size factor when morphing
    NOTEHEAD_SIZE_RATIO = 1.285714  # width/height

    STEM_WIDTH_UNIT_RATIO = 0.272727  # Ratio of stem's width to unit size
    STEM_HEIGHT_UNIT_RATIO = 4  # Ratio of stem's height to unit size

    CLEF_ZONE_WIDTH_UNIT_RATIO = 4.5406916
    CLEF_WIDTH_UNIT_RATIO = 3.2173913
    SMALL_CLEF_WIDTH_UNIT_RATIO = 2.4347826

    STAFFLINE_WIDTH_UNIT_RATIO = 0.261

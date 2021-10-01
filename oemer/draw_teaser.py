import cv2
import numpy as np

from oemer import layers


ori_img = layers.get_layer('original_image')
notes = layers.get_layer('notes')
groups = layers.get_layer('note_groups')
barlines = layers.get_layer('barlines')
clefs = layers.get_layer('clefs')
sfns = layers.get_layer('sfns')
rests = layers.get_layer('rests')

out = np.copy(ori_img).astype(np.uint8)

def draw_bbox(bboxes, color, text=None, labels=None, text_y_pos=1):
    for idx, (x1, y1, x2, y2) in enumerate(bboxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        y_pos = y1 + round((y2-y1)*text_y_pos)
        if text is not None:
            cv2.putText(out, text, (x2+2, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        else:
            cv2.putText(out, labels[idx], (x2+2, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)



draw_bbox([gg.bbox for gg in groups], color=(255, 192, 92), text="group")
draw_bbox([n.bbox for n in notes if not n.invalid], color=(194, 81, 167), labels=[str(n.label)[0] for n in notes if not n.invalid])
draw_bbox([b.bbox for b in barlines], color=(63, 87, 181), text='barline', text_y_pos=0.5)
draw_bbox([s.bbox for s in sfns if s.note_id is None], color=(90, 0, 168), labels=[str(s.label.name) for s in sfns if s.note_id is None])
draw_bbox([c.bbox for c in clefs], color=(235, 64, 52), labels=[c.label.name for c in clefs])
draw_bbox([r.bbox for r in rests], color=(12, 145, 0), labels=[r.label.name for r in rests])

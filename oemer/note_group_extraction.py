import cv2
import numpy as np
import scipy.ndimage

from oemer import layers
from oemer.inference import predict
from oemer.utils import find_closest_staffs, get_global_unit_size, get_unit_size, get_logger
from oemer.bbox import (
    get_center,
    merge_nearby_bbox,
    get_bbox,
    rm_merge_overlap_bbox,
    to_rgb_img,
    draw_bounding_boxes
)


logger = get_logger(__name__)


class NoteGroup:
    def __init__(self):
        self.id: int = None
        self.bbox: list[int] = None
        self.note_ids: list[int] = []
        self.top_note_ids: list[int] = []  # For multi-melody cases
        self.bottom_note_ids: list[int] = []  # For multi-melody cases
        self.stem_up: bool = None
        self.has_stem: bool = None
        self.all_same_type: bool = None  # All notes are solid or hollow
        self.group: int = None
        self.track: int = None

    @property
    def x_center(self):
        return (self.bbox[0] + self.bbox[2]) / 2

    def __len__(self):
        return len(self.note_ids)

    def __repr__(self):
        return f"Note Group No. {self.id} / Group: {self.group} / Track: {self.track} :(\n" \
            f"\tNote count: {len(self.note_ids)}\n" \
            f"\tStem up: {self.stem_up}\n" \
            f"\tHas stem: {self.has_stem}\n" \
            ")\n"


def group_noteheads():
    # Fetch parameters
    note_id_map = layers.get_layer('note_id')
    notehead = layers.get_layer('notehead_pred')
    stems = layers.get_layer('stems_rests_pred')

    # Extend the region of stems
    ker = np.ones((3, 2), dtype=np.uint8)
    ext_stems = cv2.dilate(stems.astype(np.uint8), ker)

    # Label each potential group region
    mix = notehead + ext_stems
    mix[mix>1] = 1
    nh_label, _ = scipy.ndimage.label(mix)
    nids = set(np.unique(note_id_map))
    if -1 in nids:
        nids.remove(-1)

    groups = {}
    for nid in nids:
        nys, nxs = np.where(note_id_map==nid)

        offset = 3
        top = np.min(nys) - offset #unit_size // 2
        bt = np.max(nys) + offset #unit_size // 2
        left = np.min(nxs)  # - unit_size // 3
        right = np.max(nxs)  # + unit_size // 3
        covered_region = nh_label[top:bt, left:right]
        labels = set(np.unique(covered_region))
        if 0 in labels:
            labels.remove(0)

        if len(labels) == 0:
            continue
        elif len(labels) > 1:
            keys = set(groups.keys())
            inter = labels.intersection(keys)
            if len(inter) == 0:
                # The first member of this group
                label = labels.pop()
            elif len(inter) == 1:
                # Already has this group
                label = inter.pop()
            else:
                # Being the middle part of the sandwitch.
                # Overlapping with multiple registered groups.
                label = inter.pop()
                tmp_g = groups[label]
                for k in inter:
                    tmp_g.extend(groups[k])
                    del groups[k]
                groups[label] = tmp_g

            for ll in labels:
                # Update the label map
                nh_label[nh_label==ll] = label
        else:
            label = labels.pop()

        if label not in groups:
            groups[label] = []
        groups[label].append(nid)

    # Remove the groups that has no noteheads attached to.
    lls = set(np.unique(nh_label))
    diff = lls.difference(groups.keys())
    for ll in diff:
        nh_label[nh_label==ll] = 0

    return groups, nh_label


def get_possible_nearby_gid(cur_note, group_map, scan_range_ratio=5):
    bbox = cur_note.bbox
    cen_x, cen_y = get_center(bbox)
    cur_gid = group_map[cen_y, cen_x]
    unit_size = get_unit_size(cen_x, cen_y)

    w = bbox[2] - bbox[0] + 4
    start_x = bbox[0] - round(w / 2)
    end_x = min(start_x+w, group_map.shape[1])
    def search(cur_y, y_bound, step):
        while True:
            if step > 0 and cur_y >= y_bound:
                break
            elif step < 0 and cur_y < y_bound:
                break
            pxs = group_map[int(cur_y), int(start_x):int(end_x)]
            gids = set(np.unique(pxs))
            if 0 in gids:
                gids.remove(0)
            if cur_gid in gids:
                gids.remove(cur_gid)

            if len(gids) > 0:
                if len(gids) > 1:
                    # Get the gid with largest overlapped region.
                    reg = []
                    for gg in gids:
                        reg.append((gg, pxs[pxs==gg].size))
                    gid = sorted(reg, key=lambda it: it[1])[-1][0]
                else:
                    gid = gids.pop()
                return gid, cur_y
            cur_y += step
        return None, None

    st1, st2 = find_closest_staffs(cen_x, cen_y)
    y_upper = min(st1.y_upper, st2.y_upper)
    y_lower = max(st1.y_lower, st2.y_lower)

    # Grid search up
    cur_y = bbox[1] - 1
    y_bound = max(cur_y - scan_range_ratio * unit_size, y_upper)
    gid_top, gty = search(cur_y, y_bound, -1)

    # Grid search down
    cur_y = bbox[3] + 1
    y_bound = min(cur_y + scan_range_ratio * unit_size, y_lower)
    gid_bt, gby = search(cur_y, y_bound, 1)

    if gid_top is not None and gid_bt is not None:
        diff_top = abs(cen_y - gty)
        diff_bt = abs(cen_y - gby)
        return gid_top if diff_top < diff_bt else gid_bt
    elif gid_top is not None:
        return gid_top
    elif gid_bt is not None:
        return gid_bt
    return None


def check_valid_new_group(ori_grp, tar_grp, group_map, max_x_diff_ratio=0.5):
    if tar_grp is None:
        return True

    def _get_box(gid):
        ys, xs = np.where(group_map==gid)
        return (np.min(xs), np.min(ys), np.max(xs), np.max(ys))

    ori_box = _get_box(ori_grp)
    tar_box = _get_box(tar_grp)
    ori_x_cen, ori_y_cen = get_center(ori_box)
    tar_x_cen, _ = get_center(tar_box)
    unit_size = get_unit_size(ori_x_cen, ori_y_cen)
    max_x_diff = unit_size * max_x_diff_ratio
    diff = abs(tar_x_cen - ori_x_cen)
    return diff < max_x_diff


def parse_stem_direction(groups, group_map, tolerance_ratio=0.2, max_x_diff_ratio=0.5):
    # Fetch parameters
    notes = layers.get_layer('notes')

    temp_result = {}
    for gp, nids in groups.items():
        gy, gx = np.where(group_map==gp)
        gbox = (np.min(gx), np.min(gy), np.max(gx), np.max(gy))
        nbox = np.array([notes[nid].bbox for nid in nids])
        nbox = (np.min(nbox[:, 0]), np.min(nbox[:, 1]), np.max(nbox[:, 2]), np.max(nbox[:, 3]))
        nh = np.mean([notes[nid].bbox[3]-notes[nid].bbox[1] for nid in nids])  # Average note height in this group
        tolerance = nh * tolerance_ratio

        gp_higher = (gbox[1] < nbox[1] - tolerance)
        gp_lower = (gbox[3] > nbox[3] + tolerance)

        if gp_higher and not gp_lower:
            # Stems up
            temp_result[gp] = True
            for nid in nids:
                notes[nid].stem_up = True
            continue
        elif not gp_higher and gp_lower:
            # Stems down
            temp_result[gp] = False
            for nid in nids:
                notes[nid].stem_up = False
            continue

        # Contains both direction or has no stems, indicating there are two different melody lines or 
        # it's a single whole note.
        if len(nids) == 1:
            nid = nids[0]
            new_group = get_possible_nearby_gid(notes[nid], group_map)
            if (new_group is not None) and check_valid_new_group(gp, new_group, group_map, max_x_diff_ratio):
                if new_group in temp_result:
                    notes[nid].stem_up = temp_result[new_group]

                # Update groups and group_map
                group_map = np.where(group_map==gp, new_group, group_map)
                groups[new_group].append(nid)
                old_gp_nidx = groups[gp].index(nid)
                del groups[gp][old_gp_nidx]

    groups = {gp: nids for gp, nids in groups.items() if len(nids) > 0}
    return groups, group_map


def check_group(group):
    notes = layers.get_layer('notes')

    if group.has_stem and group.stem_up is not None:
        # Check stem's height
        box = group.bbox
        ny_bound = np.array([(notes[nid].bbox[1], notes[nid].bbox[3]) for nid in group.note_ids])
        if group.stem_up:
            diff = abs(box[1] - np.min(ny_bound[:, 0]))
        else:
            diff = abs(box[3] - np.max(ny_bound[:, 1]))
        unit_size = get_unit_size(*get_center(box))
        if diff < unit_size:
            for nid in group.note_ids:
                notes[nid].invalid = True
            return False
    return True


def gen_groups(groups, group_map):
    # Fetch parameters
    notes = layers.get_layer('notes')

    global grp_img
    grp_img = np.copy(group_map)
    grp_img = to_rgb_img(grp_img)

    ngs = []
    new_map = np.zeros_like(group_map) - 1
    idx = 0
    for gid, nids in groups.items():
        ng = NoteGroup()
        ng.id = idx
        ng.note_ids = nids
        gy, gx = np.where(group_map==gid)
        gbox = (np.min(gx), np.min(gy), np.max(gx), np.max(gy))
        nbox = np.array([notes[nid].bbox for nid in nids])
        nbox = (np.min(nbox[:, 0]), np.min(nbox[:, 1]), np.max(nbox[:, 2]), np.max(nbox[:, 3]))

        cv2.rectangle(grp_img, (gbox[0], gbox[1]), (gbox[2], gbox[3]), (255, 0, 0), 2)
        cv2.rectangle(grp_img, (nbox[0], nbox[1]), (nbox[2], nbox[3]), (0, 0, 255), 2)

        ng.bbox = gbox
        for nid in nids:
            notes[nid].note_group_id = idx

        if notes[nids[0]].stem_up is None:
            # Stems are at both side, or no stems.
            nh = np.mean([notes[nid].bbox[3]-notes[nid].bbox[1] for nid in nids])  # Average note height in this group
            g_height = gbox[3] - gbox[1]
            n_height = nbox[3] - nbox[1]
            if abs(g_height-n_height) > nh // 5:
                #assert len(nids) > 1, nids
                ng.has_stem = True
            else:
                ng.has_stem = False
        elif notes[nids[0]].stem_up:
            ng.stem_up = True
            ng.has_stem = True
        else:
            ng.stem_up = False
            ng.has_stem = True

        n_types = [notes[nid].label for nid in nids]
        ng.all_same_type = all(nt==n_types[0] for nt in n_types)

        # Do some post check
        tar_track = notes[nids[0]].track
        tar_group = notes[nids[0]].group
        same_track = all(notes[nid].track==tar_track for nid in nids)
        same_group = all(notes[nid].group==tar_group for nid in nids)
        if not (same_track and same_group):
            y_mass_center = (gbox[1] + gbox[3]) / 2
            x_mass_center = (gbox[0] + gbox[2]) / 2
            st, _ = find_closest_staffs(x_mass_center, y_mass_center)
            tar_track = st.track
            tar_group = st.group
            for nid in nids:
                notes[nid].track = st.track
                notes[nid].group = st.group

        ng.track = tar_track
        ng.group = tar_group

        new_map[group_map==gid] = idx
        ngs.append(ng)
        idx += 1
    return ngs, new_map


def post_check_groups(groups):
    # Fetch parameters
    notes = layers.get_layer('notes')

    for grp in groups:
        if len(grp.note_ids) != 2:
            # Currently only supports to separate mis-grouping notes
            # that contains only two notes in current group.
            continue


def extract():
    # Start process
    logger.debug("Grouping noteheads")
    groups, group_map = group_noteheads()

    logger.debug("Analyzing stem direction")
    groups, group_map = parse_stem_direction(groups, group_map)

    logger.debug("Instanitiating note groups")
    groups, group_map = gen_groups(groups, group_map)

    logger.debug("Post check notes in groups")

    return groups, group_map


def predict_symbols():
    pred = layers.get_layer('celfs_keys_pred')  # sfn -> sharp, flat, natural
    #pred = layers.get_layer('stems_rests_pred')
    bboxes = get_bbox(pred)
    bboxes = merge_nearby_bbox(bboxes, 15)
    bboxes = rm_merge_overlap_bbox(bboxes)

    img = np.ones(pred.shape+(3,), dtype=np.uint8) * 255
    idx = np.where(pred>0)
    img[idx[0], idx[1]] = 0
    for box in bboxes:
        region = pred[box[1]:box[3], box[0]:box[2]]
        region[region>0] = 255
        pp = predict(region, "sfn")
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(img, pp, (box[2]+2, box[3]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1)
    return img


def draw_notes(notes, ori_img):
    img = ori_img.copy()
    img = np.array(img)
    for note in notes:
        x1, y1, x2, y2 = note.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if note.has_dot:
            cv2.putText(img, "DOT", (x2+2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return img


if __name__ == "__main__":
    notes = layers.get_layer('notes')
    symbols = layers.get_layer('symbols_pred')
    note_id_map = layers.get_layer('note_id')
    notehead = layers.get_layer('notehead_pred')
    stems_rests = layers.get_layer('stems_rests_pred')
    ori_img = layers.get_layer('original_image')

    img = np.ones(notehead.shape+(3,)) * 255
    idx = np.where(stems_rests>0)
    img[idx[0], idx[1]] = 0

    unit_size = get_global_unit_size()
    logger.info("Grouping noteheads")
    a_groups, a_map = group_noteheads()
    logger.info("Analyzing stem direction")
    b_groups, b_map = parse_stem_direction(a_groups, a_map)
    logger.info("Instanitiating note groups")
    groups, c_map = gen_groups(b_groups, b_map)

    bboxes = [g.bbox for g in groups]
    out = draw_bounding_boxes(bboxes, notehead)

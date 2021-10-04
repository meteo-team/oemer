import math
import enum
from typing import List, Mapping
import xml.etree.ElementTree as ET
from xml.dom import minidom
from xml.etree.ElementTree import Element, SubElement

import numpy as np

from oemer import layers
from oemer.symbol_extraction import Barline, Clef, Sfn, Rest, SfnType, ClefType, RestType
from oemer.note_group_extraction import NoteGroup
from oemer.notehead_extraction import NoteHead, NoteType
from oemer.utils import get_global_unit_size, get_logger, get_total_track_nums


logger = get_logger(__name__)

DIVISION_PER_QUATER = 16  # Equal to length of 64th note.

G_CLEF_POS_TO_PITCH = ['D', 'E', 'F', 'G', 'A', 'B', 'C']
F_CLEF_POS_TO_PITCH = ['F', 'G', 'A', 'B', 'C', 'D', 'E']
SHARP_KEY_ORDER = ['F', 'C', 'G', 'D', 'A', 'E']
FLAT_KEY_ORDER = ['B', 'E', 'A', 'D', 'G', 'C']

NOTE_TYPE_TO_RHYTHM = {
    NoteType.WHOLE: {"name": "whole", "duration": DIVISION_PER_QUATER * 4},
    NoteType.HALF: {"name": "half", "duration": DIVISION_PER_QUATER * 2},
    NoteType.HALF_OR_WHOLE: {"name": "half", "duration": DIVISION_PER_QUATER * 2},
    NoteType.QUARTER: {"name": "quarter", "duration": DIVISION_PER_QUATER},
    NoteType.EIGHTH: {"name": "eighth", "duration": DIVISION_PER_QUATER // 2},
    NoteType.SIXTEENTH: {"name": "16th", "duration": DIVISION_PER_QUATER // 4},
    NoteType.THIRTY_SECOND: {"name": "32nd", "duration": DIVISION_PER_QUATER // 8},
    NoteType.SIXTY_FOURTH: {"name": "64th", "duration": DIVISION_PER_QUATER // 16}
}

REST_TYPE_TO_DURATION = {
    RestType.QUARTER: DIVISION_PER_QUATER,
    RestType.EIGHTH: DIVISION_PER_QUATER // 2,
    RestType.SIXTEENTH: DIVISION_PER_QUATER // 4,
    RestType.THIRTY_SECOND: DIVISION_PER_QUATER // 8,
    RestType.SIXTY_FOURTH: DIVISION_PER_QUATER // 16,
    RestType.WHOLE_HALF: DIVISION_PER_QUATER * 2,
    RestType.WHOLE: DIVISION_PER_QUATER * 4,
    RestType.HALF: DIVISION_PER_QUATER * 2
}

DURATION_TO_REST_TYPE = {v: k for k, v in REST_TYPE_TO_DURATION.items()}


class Key(enum.Enum):
    C_MAJOR = 0  # Same as A-minor
    G_MAJOR = 1  # Same as E-minor
    D_MAJOR = 2  # Same as B-minor
    A_MAJOR = 3  # Same as F#-minor
    E_MAJOR = 4  # Same as C#-minor
    B_MAJOR = 5  # Same as G#-minor
    F_SHARP_MAJOR = 6  # Same as D#-minor
    F_MAJOR = -1  # Same as D-minor
    B_FLAT_MAJOR = -2  # Same as G-minor
    E_FLAT_MAJOR = -3  # Same as C-minor
    A_FLAT_MAJOR = -4  # Same as F-minor
    D_FLAT_MAJOR = -5  # Same as Bb-minor
    G_FLAT_MAJOR = -6  # Same as Eb-minor


class Voice:
    def __init__(self):
        self.id: int = None
        self.note_ids: list[int] = []
        self.stem_up: bool = None
        self.group_id: int = None
        self.x_center: float = None
        self.label: NoteType = None
        self.has_dot: bool = None
        self.group: int = None
        self.track: int = None
        self.duration: int = None
        self.rhythm_name: str = None

    def init(self):
        notes = layers.get_layer('notes')

        # Determine the label
        labels = [notes[nid].label for nid in self.note_ids]
        ll_count = {ll: 0 for ll in set(labels)}
        for ll in labels:
            ll_count[ll] += 1
        ll_count = sorted(ll_count.items(), key=lambda it: it[1], reverse=True)
        self.label = ll_count[0][0]

        # Determine the dots
        dots = [notes[nid].has_dot for nid in self.note_ids]
        t_count = sum(dots)
        f_count = len(dots) - t_count
        self.has_dot = True if t_count > f_count else False

        # Update the label of notes
        for nid in self.note_ids:
            if notes[nid].label != self.label:
                notes[nid].force_set_label(self.label)
            notes[nid].has_dot = self.has_dot

        self.rhythm_name = NOTE_TYPE_TO_RHYTHM[self.label]['name']
        self.duration = NOTE_TYPE_TO_RHYTHM[self.label]['duration']
        if self.has_dot:
            self.duration = round(self.duration * 1.5)

    def __repr__(self):
        return f"Voice {self.id}\n" \
            f"\tGroup: {self.group} / Track: {self.track} / Rhythm: {self.rhythm_name} / Duration: {self.duration}\n" \
            f"\tNotes: {len(self.note_ids)} / Note Group: {self.group_id} / Dot: {self.has_dot}\n"


class Measure:
    def __init__(self):
        self.symbols = []  # List of symbols
        self.double_barline: bool = None
        self.has_clef: bool = False
        self.clefs: list[Clef] = []
        self.voices: list[NoteGroup] = []
        self.sfns: list[Sfn] = []
        self.rests: list[Rest] = []
        self.number: int = None
        self.at_beginning: bool = None
        self.group: int = None

        self.time_slots: list[object] = []
        self.slot_duras: np.ndarray = None

    def add_symbols(self, symbols):
        self.symbols.extend(symbols)
        self.symbols = sorted(self.symbols, key=lambda s: s.x_center)
        for sym in symbols:
            if isinstance(sym, Voice):
                self.voices.append(sym)
            elif isinstance(sym, Clef):
                self.clefs.append(sym)
                self.has_clef = True
            elif isinstance(sym, Sfn):
                self.sfns.append(sym)
            elif isinstance(sym, Rest):
                self.rests.append(sym)
            else:
                raise ValueError(f"Invalid instance type: {type(sym)}")
        self.voices = sorted(self.voices, key=lambda s: s.x_center)
        self.clefs = sorted(self.clefs, key=lambda s: s.x_center)
        self.sfns = sorted(self.sfns, key=lambda s: s.x_center)
        self.rests = sorted(self.rests, key=lambda s: s.x_center)

    def has_key(self):
        total_tracks = get_total_track_nums()
        start_idx = total_tracks if self.at_beginning else 0
        syms =  self.symbols[start_idx:start_idx+total_tracks]
        return all(isinstance(sym, Sfn) for sym in syms)

    def get_key(self):
        if len(self.sfns) == 0:
            return Key(0)

        track_nums = get_total_track_nums()

        if self.has_key():
            if self.at_beginning:
                # The first <track_nums> elements should be clefs
                start_idx = track_nums
            else:
                start_idx = 0
            end_idx = track_nums * 7 + 4  # There are at most 6 sharps/flats. Some tolerance are added.
            end_idx = min(end_idx, len(self.symbols))
        else:
            return Key(0)

        # Prepare data
        sfns_cands = []  # Candidates of key
        for i in range(start_idx, end_idx):
            sym = self.symbols[i]
            if not isinstance(sym, Sfn):
                break
            sfns_cands.append(sym)

        # Count occurance
        sfn_counts = [0 for _ in range(track_nums)]
        for sfn in sfns_cands:
            sfn_counts[sfn.track] += 1

        # Check validity
        all_same = all(ss.label==sfns_cands[0].label for ss in sfns_cands)  # All tracks have the same label.
        all_equal = all(cc==sfn_counts[0] for cc in sfn_counts)  # All tracks have the same number of sfns.

        if not all_equal:
            logger.warning("The number of key symbols are not all the same for every track!")

        sfn_label = sfns_cands[0].label
        if not all_same:
            # Count the most occurance.
            counter = {SfnType.FLAT: 0, SfnType.SHARP: 0, SfnType.NATURAL: 0}
            for sfn in sfns_cands:
                counter[sfn.label] += 1
            counter = sorted(counter.items(), key=lambda s: s[1], reverse=True)
            if counter[0][0] == SfnType.NATURAL:
                # Swap the first and second candidates when the first sfn type is natural.
                counter[0], counter[1] = counter[1], counter[0]

            if counter[0][0] == SfnType.FLAT:
                # Flat and sharp/natural is assumed will not being confused by the model.
                sfn_label = SfnType.FLAT
            elif counter[0][0] == SfnType.SHARP:
                if counter[1][0] == SfnType.FLAT:
                    sfn_label = SfnType.SHARP
                elif counter[0][1] > counter[1][1]:
                    # Sharp and natural are the most error-prone for prediction.
                    sfn_label = SfnType.SHARP
                else:
                    # The most special case that number of sharp and natural are the same,
                    # which means the key indeed contains natural.
                    # TODO: should allow natural key
                    sfn_label = SfnType.SHARP

        count = round(sum(sfn_counts) / track_nums)
        if sfn_label == SfnType.FLAT:
            count *= -1

        # Update the sfn state
        for sfn in sfns_cands:
            sfn.is_key = True

        return Key(count)

    def get_track_clef(self):
        track_nums = get_total_track_nums()
        if self.at_beginning or self.double_barline:
            clefs = []
            for track in range(track_nums):
                clef = [clef for clef in self.clefs if clef.track == track]
                if clef:
                    clefs.append(clef[0])
                else:
                    clef = Clef()
                    clef.track = track
                    clef.group = self.group
                    clef.label = ClefType(track%2 + 1)
                    clefs.append(clef)
            return clefs
        return [None for _ in range(track_nums)]

    def align_symbols(self):
        track_nums = get_total_track_nums()
        unit_size = get_global_unit_size()
        time_slots = []
        corr_sidx = []
        last_x = None
        for idx, sym in enumerate(self.symbols):
            if isinstance(sym, Clef):
                if self.at_beginning:
                    continue
                # TODO: process non-beginning case.
                continue
            elif isinstance(sym, Sfn):
                continue

            if last_x is None:
                last_x = sym.x_center
                time_slots.append([sym])
                corr_sidx.append(idx)
            else:
                if abs(sym.x_center - last_x) < unit_size:
                    time_slots[-1].append(sym)
                else:
                    time_slots.append([sym])
                    corr_sidx.append(idx)
                    last_x = sym.x_center

        # Collect necessary informations for determining rhythm mismatch.
        multi_track_idx = []
        track_duras = np.zeros((len(time_slots), track_nums), dtype=np.uint16)
        for idx, slot in enumerate(time_slots):
            has_multi = len(set(sym.track for sym in slot)) > 1
            if has_multi:
                multi_track_idx.append(idx)

            duras = [[] for _ in range(track_nums)]
            for sym in slot:
                dura = get_duration(sym)
                duras[sym.track].append(dura)
            for track, dura in enumerate(duras):
                du = min(dura) if dura else 0
                track_duras[idx, track] = du

        # Decide whether to keep processing
        if track_nums == 1:
            self.time_slots = time_slots
            self.slot_duras = track_duras
            return
        assert track_nums == 2, track_nums

        # Start adjusting rhythms
        diff = 0
        lead_track = None
        add_idx = None
        added = 0
        solved = True
        new_track_duras = np.zeros((track_duras.shape[0], track_duras.shape[1]+2))
        new_track_duras[:, :2] = np.copy(track_duras)

        def modify(add_idx, diff, lead_track, added):
            pos_dura = track_duras[add_idx, 1-lead_track]
            if pos_dura != 0:
                # Extend the length of the original symbol
                for sym in time_slots[add_idx]:
                    if sym.track == 1-lead_track:
                        ori_label = sym.label
                        ori_dot = sym.has_dot
                        extend_symbol_length(sym, diff+pos_dura)
                        logger.debug(f"Before: {ori_label}, Dot: {ori_dot}" \
                                    f" / After: {sym.label}, Dot: {sym.has_dot}")
                new_track_duras[add_idx, 1-lead_track] = diff+pos_dura
            else:
                # Add rest to balance the rhythm
                rest = get_rest(diff)
                rest.track = 1 - lead_track
                rest.group = self.group
                x_center = time_slots[add_idx][0].x_center
                rest.bbox = (x_center, 0, x_center, 0)
                insert_idx = corr_sidx[idx] + added
                self.symbols.insert(insert_idx, rest)
                time_slots[add_idx].insert(0, rest)
                added += 1
                logger.debug(f"Add: {diff}, {rest}, {insert_idx}, {add_idx}, {1-lead_track}")
                new_track_duras[add_idx, 1-lead_track] = diff
            return added

        for idx, duras in enumerate(track_duras):
            t1, t2 = duras
            if (t1 > 0) and (t2 > 0):
                if diff > 0:
                    added = modify(add_idx, diff, lead_track, added)
                if t1 > t2:
                    diff = t1 - t2
                    lead_track = 0
                else:
                    diff = t2 - t1
                    lead_track = 1
                add_idx = idx
                solved = True
            elif t1 > 0:
                if lead_track == 0:
                    diff += t1
                elif diff >= t1:
                    diff -= t1
                else:
                    if 0 < diff < t1:
                        diff = t1 - diff
                    else:
                        diff = t1
                    add_idx = idx
                    lead_track = 0
                solved = False
            elif t2 > 0:
                if lead_track == 1:
                    diff += t2
                elif diff >= t2:
                    diff -= t2
                else:
                    if 0 < diff < t2:
                        diff = t2 - diff
                    else:
                        diff = t2
                    add_idx = idx
                    lead_track = 1
                solved = False

            new_track_duras[idx, 2] = diff
            new_track_duras[idx, 3] = lead_track

        if not solved and diff > 0:
            added = modify(add_idx, diff, lead_track, added)

        self.time_slots = time_slots
        self.slot_duras = new_track_duras[:, :2]
        return time_slots, track_duras, new_track_duras

    def get_time_slot_dura(self, x_center):
        for idx, slot in enumerate(self.time_slots[:-1]):
            if slot[0].x_center <= x_center < self.time_slots[idx+1][0].x_center:
                return idx, self.slot_duras[idx]
        return len(self.time_slots)-1, self.slot_duras[-1]

    def __repr__(self):
        return f"Measure: {self.number} / Symbols: {len(self.symbols)}" \
            f" / Has clef: {self.has_clef}"


class Action:

    class Context:
        key: Key = None
        clefs: List[Clef] = []
        sfn_state: Mapping[str, Sfn] = {chr(ord('A')+i):None for i in range(7)}

    ctx = Context()

    def __init__(self):
        pass

    def perform(self, **kwargs) -> Element:
        raise NotImplementedError

    @classmethod
    def init_sfn_state(cls):
        cls.ctx.sfn_state = {chr(ord('A')+i):None for i in range(7)}
        if cls.ctx.key.value > 0:
            for sfn_name in SHARP_KEY_ORDER[:cls.ctx.key.value]:
                cls.ctx.sfn_state[sfn_name] = SfnType.SHARP
        elif cls.ctx.key.value < 0:
            for sfn_name in FLAT_KEY_ORDER[:-cls.ctx.key.value]:
                cls.ctx.sfn_state[sfn_name] = SfnType.FLAT

    @classmethod
    def clear(cls):
        cls.ctx.key = None
        cls.ctx.clefs = []
        cls.ctx.sfn_state = {chr(ord('A')+i):None for i in range(7)}


class KeyChange(Action):
    def __init__(self, key, **kwargs):
        super().__init__(**kwargs)
        self.key = key

    def perform(self, parent_elem=None) -> Element:
        self.ctx.key = self.key
        self.init_sfn_state()
        elem = decode_key(self.key)
        if parent_elem is not None:
            parent_elem.append(elem)
        return elem


class ClefChange(Action):
    def __init__(self, clef, **kwargs):
        super().__init__(**kwargs)
        self.clef = clef

    def perform(self, parent_elem=None) -> Element:
        self.ctx.clefs[self.clef.track] = self.clef
        elem = decode_clef(self.clef)
        if parent_elem is not None:
            parent_elem.append(elem)
        return elem


class AddNote(Action):
    def __init__(self, note: NoteHead, chord=False, voice=1, **kwargs):
        super().__init__(**kwargs)
        self.note = note
        self.chord = chord
        self.voice = voice

    def perform(self, parent_elem=None):
        clef_type = self.ctx.clefs[self.note.track].label
        chroma = get_chroma_pitch(self.note.staff_line_pos, clef_type)
        cur_sfn = self.ctx.sfn_state[chroma]
        if (self.note.sfn is not None) and (cur_sfn != self.note.sfn):
            self.ctx.sfn_state[chroma] = self.note.sfn
        else:
            self.note.sfn = cur_sfn
        elem = decode_note(self.note, clef_type, self.chord, self.voice)
        if elem is not None and parent_elem is not None:
            parent_elem.append(elem)
        return elem


class AddRest(Action):
    def __init__(self, rest, **kwargs):
        super().__init__(**kwargs)
        self.rest = rest

    def perform(self, parent_elem=None):
        elem = decode_rest(self.rest)
        if parent_elem is not None:
            parent_elem.append(elem)
        return elem


class AddBackup(Action):
    def __init__(self, dura, **kwargs):
        super().__init__(**kwargs)
        self.dura = dura

    def perform(self, parent_elem=None):
        elem = decode_backup(self.dura)
        if parent_elem is not None:
            parent_elem.append(elem)
        return elem


class AddForward(Action):
    def __init__(self, dura: int, **kwargs):
        super().__init__(**kwargs)
        self.dura = dura

    def perform(self, parent_elem=None):
        elem = decode_forward(self.dura)
        if parent_elem is not None:
            parent_elem.append(elem)
        return elem


class AddMeasure(Action):
    def __init__(self, measure: Measure, **kwargs):
        super().__init__(**kwargs)
        self.measure = measure

    def perform(self, parent_elem=None):
        self.init_sfn_state()
        elem = Element('measure', attrib={'number': str(self.measure.number)})
        if parent_elem is not None:
            parent_elem.append(elem)
        return elem


class AddInit(Action):
    def __init__(self, measure: Measure, **kwargs):
        super().__init__(**kwargs)
        assert measure.at_beginning
        self.measure = measure

    def perform(self, parent_elem=None):
        self.ctx.key = self.measure.get_key()
        self.ctx.clefs = self.measure.get_track_clef()
        self.init_sfn_state()

        elem = Element('measure', attrib={'number': str(self.measure.number)})
        attr = SubElement(elem, 'attributes')
        div = SubElement(attr, 'divisions')
        div.text = str(DIVISION_PER_QUATER)
        key = decode_key(self.measure.get_key())
        key = list(key)[0]
        attr.append(key)
        staves = SubElement(attr, 'staves')
        staves.text = str(get_total_track_nums())

        track_clefs = self.measure.get_track_clef()
        for clef in track_clefs:
            clef_elem = decode_clef(clef)
            clef_elem = list(clef_elem)[0]
            attr.append(clef_elem)

        if parent_elem is not None:
            parent_elem.append(elem)
        return elem


class MusicXMLBuilder:
    def __init__(self, title=None):
        self.measures: dict[int, list[Measure]] = {}
        self.actions: list[Action] = []
        self.title: str = title

    def build(self):
        # Fetch parameters
        notes = layers.get_layer('notes')

        voices = get_voices()
        group_container = sort_symbols(voices)
        self.gen_measures(group_container)

        Action.clear()
        first_measure = self.measures[0][0]
        self.actions.append(AddInit(first_measure))

        cur_key = first_measure.get_key()
        cur_clefs = first_measure.get_track_clef()
        total_tracks = len(cur_clefs)
        for grp, measures in self.measures.items():
            for idx, measure in enumerate(measures):
                self.actions.append(AddMeasure(measure))
                last_tidx = 0
                last_dura = 0
                last_pos = 0
                total_track_duras = [0 for _ in range(total_tracks)]
                min_dura_added = [False for _ in range(total_tracks)]
                for sidx, sym in enumerate(measure.symbols):
                    track = sym.track
                    if isinstance(sym, Clef):
                        if cur_clefs[track].label != sym.label:
                            self.actions.append(ClefChange(sym))
                            cur_clefs[track] = sym
                    elif isinstance(sym, Sfn):
                        if measure.has_key():
                            key = measure.get_key()
                            if key != cur_key:
                                self.actions.append(KeyChange(key))
                                cur_key = key
                    else:
                        # Adjust beat position first
                        tidx, min_duras = measure.get_time_slot_dura(sym.x_center)
                        min_dura = min_duras[track]
                        dura = get_duration(sym)
                        if tidx == last_tidx and last_dura > 0:
                            self.actions.append(AddBackup(int(last_dura)))
                        else:
                            min_dura_added = [False for _ in range(total_tracks)]
                            diff = last_pos - total_track_duras[track]
                            if diff > 0 and diff != last_dura:
                                self.actions.append(AddBackup(int(diff)))

                        # Update some inner state according to voice number.
                        if (dura == min_dura) and (not min_dura_added[track]):
                            # It's the first voice.
                            total_track_duras[track] += min_dura
                            min_dura_added[track] = True
                            voice_one = True
                        else:
                            # The second voice.
                            voice_one = False

                        # Add the symbol
                        if isinstance(sym, Rest):
                            self.actions.append(AddRest(sym))
                            last_pos = total_track_duras[track]
                            last_dura = dura
                            last_tidx = tidx
                        elif isinstance(sym, Voice):
                            # # Adjust invalid sfn state of ntoes.
                            # for nid in sym.note_ids:
                            #     if notes[nid].sfn == SfnType.FLAT and cur_key.value > 0:
                            #         notes[nid].sfn = SfnType.NATURAL
                            #     elif notes[nid].sfn == SfnType.SHARP and cur_key.value < 0:
                            #         notes[nid].sfn = SfnType.NATURAL

                            # Add action to the queue
                            if sym.duration == min_dura and voice_one:
                                self.actions.append(AddNote(notes[sym.note_ids[0]]))
                                for nid in sym.note_ids[1:]:
                                    self.actions.append(AddNote(notes[nid], chord=True))
                                last_pos = total_track_duras[track]
                                last_dura = dura
                                last_tidx = tidx
                            else:
                                # Second voice in the track. Need special process.
                                self.actions.append(AddNote(notes[sym.note_ids[0]], voice=2))
                                for nid in sym.note_ids[1:]:
                                    self.actions.append(AddNote(notes[nid], chord=True, voice=2))
                                cur_pos = total_track_duras[track] + sym.duration
                                if min_dura_added[track]:
                                    # Check whether the min dura has been added or not.
                                    # If yes, need to backup extra duration.
                                    cur_pos -= min_dura
                                diff = last_pos - cur_pos
                                if diff > 0:
                                    self.actions.append(AddForward(int(diff)))
                                elif diff < 0:
                                    self.actions.append(AddBackup(int(-diff)))

    def gen_measures(self, group_container):
        num = 1  # Measure count starts from 1 for MusicXML
        for grp, insts in group_container.items():
            self.measures[grp] = []
            buffer = []
            at_beginning = True
            double_barline = False
            for inst in insts:
                if isinstance(inst, Barline):
                    if len(buffer) == 0:
                        # Double barline
                        double_barline = True
                    else:
                        mm = gen_measure(buffer, grp, num, at_beginning, double_barline)
                        self.measures[grp].append(mm)

                        num += 1
                        buffer = []
                        at_beginning = False
                        double_barline = False
                    continue
                buffer.append(inst)

            if buffer:
                # Clear out buffer
                mm = gen_measure(buffer, grp, num, at_beginning, double_barline)
                self.measures[grp].append(mm)

    def to_musicxml(self, tempo=90):
        score = Element('score-partwise', attrib={'version': '4.0'})
        work = build_work(self.title)
        iden = build_identity()
        part_list = build_part_list()
        score.extend([work, iden, part_list])
        part = SubElement(score, 'part', attrib={'id': 'P1'})
        sound = SubElement(part, "sound", attrib={"tempo": str(tempo)})

        measure = None
        for action in self.actions:
            if isinstance(action, AddInit) or isinstance(action, AddMeasure):
                measure = action.perform(parent_elem=part)
            else:
                action.perform(parent_elem=measure)

        # Construct DOCTYPE header
        mxl_str = pretty_xml(score)
        doctype = b''\
            b'<!DOCTYPE score-partwise PUBLIC' \
            b' "-//Recordare//DTD MusicXML 4.0 Partwise//EN"' \
            b' "http://www.musicxml.org/dtds/partwise.dtd">'
        mxl_list = mxl_str.split(b'?>')
        mxl_list.insert(1, doctype)
        mxl_list[0] += b'?>\n'
        mxl_str = b"".join(mxl_list)
        return mxl_str


def gen_measure(buffer, grp, num, at_beginning=False, double_barline=False):
    mm = Measure()
    mm.add_symbols(buffer)
    mm.double_barline = double_barline
    mm.number = num
    mm.at_beginning = at_beginning
    mm.group = grp
    mm.get_key()  # Initialize internal states.
    mm.align_symbols()
    return mm


def get_voices():
    # Fetch parameters
    groups = layers.get_layer('note_groups')
    notes = layers.get_layer('notes')

    voices = []
    def add_voice(grp, nids, stem_up):
        nids = [nid for nid in nids if not notes[nid].invalid]
        if len(nids) == 0:
            return
        vc = Voice()
        vc.group = grp.group
        vc.group_id = grp.id
        vc.track = grp.track
        vc.note_ids = nids
        vc.stem_up = stem_up
        vc.x_center = grp.x_center
        vc.init()

        voices.append(vc)

    for group in groups:
        if group.stem_up is None and group.has_stem:
            add_voice(group, group.top_note_ids, True)
            add_voice(group, group.bottom_note_ids, False)
        else:
            add_voice(group, group.note_ids, group.stem_up)

    for idx, voice in enumerate(voices):
        voice.id = idx
    return voices


def get_duration(sym):
    if isinstance(sym, Voice):
        return sym.duration

    assert isinstance(sym, Rest), sym
    dura = REST_TYPE_TO_DURATION[sym.label]
    if sym.has_dot:
        dura = round(dura * 1.5)
    return dura


def sort_symbols(voices):
    barlines = layers.get_layer('barlines')
    rests = layers.get_layer('rests')
    clefs = layers.get_layer('clefs')
    sfns = layers.get_layer('sfns')

    # Assign symbols to the coressponding group
    group_container = {}
    def sort_group(insts):
        for inst in insts:
            if inst.group not in group_container:
                group_container[inst.group] = []
            group_container[inst.group].append(inst)

    sort_group(voices)
    sort_group(barlines)
    sort_group(rests)
    sort_group(clefs)
    sort_group([sfn for sfn in sfns if sfn.note_id is None])  # Exclude accidental

    # Sort by their x-center
    for k in group_container:
        ll = group_container[k]
        ll = sorted(ll, key=lambda s: s.x_center)
        group_container[k] = ll

    # Sort by group
    temp_g = sorted(group_container.items(), key=lambda ele: ele[0])
    group_container = {ele[0]: ele[1] for ele in temp_g}
    return group_container


def get_label_by_dura(duration, mapping):
    min_diff = 9999999
    tar_label = None
    for label, dura in mapping.items():
        diff = duration - dura
        if diff >= 0 and diff < min_diff:
            min_diff = diff
            tar_label = label
    has_dot = False
    if min_diff > 0:
        dura = mapping[tar_label]
        #assert dura // 2 == min_diff, f"{min_diff}, {duration}"
        has_dot = True
    return tar_label, has_dot


def get_rest(duration):
    rest = Rest()
    tar_label, has_dot = get_label_by_dura(duration, REST_TYPE_TO_DURATION)
    rest.label = tar_label
    rest.has_dot = has_dot
    return rest


def get_chroma_pitch(pos, clef_type):
    order = G_CLEF_POS_TO_PITCH if clef_type == ClefType.G_CLEF else F_CLEF_POS_TO_PITCH
    pos = int(pos)
    return order[pos%7] if pos >= 0 else order[pos%-7]


def extend_symbol_length(symbol, duration):
    notes = layers.get_layer('notes')
    if isinstance(symbol, Voice):
        mapping = {k: v['duration'] for k, v in NOTE_TYPE_TO_RHYTHM.items()}
        tar_label, has_dot = get_label_by_dura(duration, mapping)
        symbol.label = tar_label
        symbol.has_dot = has_dot
        symbol.duration = duration
        symbol.rhythm_name = NOTE_TYPE_TO_RHYTHM[tar_label]['name']
        for nid in symbol.note_ids:
            notes[nid].force_set_label(tar_label)
            notes[nid].has_dot = has_dot
    elif isinstance(symbol, Rest):
        tar_label, has_dot = get_label_by_dura(duration, REST_TYPE_TO_DURATION)
        symbol.label = tar_label
        symbol.has_dot = has_dot


def gen_measures(group_container):
    measures = {}
    num = 1  # Measure count starts from 1 for MusicXML
    for grp, insts in group_container.items():
        measures[grp] = []
        buffer = []
        at_beginning = True
        double_barline = False
        for inst in insts:
            if isinstance(inst, Barline):
                if len(buffer) == 0:
                    # Double barline
                    double_barline = True
                else:
                    mm = Measure()
                    mm.add_symbols(buffer)
                    mm.double_barline = double_barline
                    mm.number = num
                    mm.get_key()  # Initialize internal states.
                    mm.at_beginning = at_beginning
                    mm.group = grp
                    measures[grp].append(mm)

                    num += 1
                    buffer = []
                    at_beginning = False
                    double_barline = False
                continue
            buffer.append(inst)
    return measures


def decode_note(note, clef_type, is_chord=False, voice=1) -> Element:
    if note.invalid:
        return None

    # Element order matters!!
    elem = Element('note')

    # Chord
    if is_chord:
        elem.append(Element('chord'))

    # Pitch
    pitch = SubElement(elem, 'pitch')
    step = SubElement(pitch, 'step')
    alter = SubElement(pitch, 'alter')
    octave = SubElement(pitch, 'octave')
    alter.text = '0'
    pos = int(note.staff_line_pos)
    if clef_type == ClefType.G_CLEF:
        order = G_CLEF_POS_TO_PITCH
        oct_offset = 4
        pitch_offset = 1
    else:
        order = F_CLEF_POS_TO_PITCH
        oct_offset = 2
        pitch_offset = 3
    step.text = order[pos%7] if pos >= 0 else order[pos%-7]
    if pos - pitch_offset >= 0:
        octave.text = str(math.floor((pos + pitch_offset) / 7) + oct_offset)
    else:
        octave.text  = str(-math.ceil((pos + pitch_offset) / -7) + oct_offset)
    if note.sfn is not None:
        if note.sfn == SfnType.SHARP:
            alter.text = '1'
        elif note.sfn == SfnType.FLAT:
            alter.text = '-1'

    # Check the pitch is within A0~C8
    if (int(octave.text) < 0 or int(octave.text) > 8) \
            or (int(octave.text) == 0 and step.text != "A") \
            or (int(octave.text) == 8 and step.text != "C"):
        return None

    # Rhythm type
    dura = SubElement(elem, 'duration')
    dura.text = str(NOTE_TYPE_TO_RHYTHM[note.label]['duration'])
    rhy = SubElement(elem, 'type')
    rhy.text = NOTE_TYPE_TO_RHYTHM[note.label]['name']

    # Dot
    if note.has_dot:
        du = int(dura.text)
        dura.text = str(round(du*1.5))
        elem.append(Element('dot'))

    # Stem direction
    stem = SubElement(elem, 'stem')
    stem.text = "up" if note.stem_up else "down"

    # Track
    track = SubElement(elem, 'staff')
    track.text = str(note.track + 1)  # Start from one for MusicXML
    voi = SubElement(elem, "voice")
    voi.text = str(voice)

    return elem


def decode_rest(rest) -> Element:
    elem = Element('note')
    SubElement(elem, 'rest', attrib={'measure': 'yes'})
    dura = SubElement(elem, 'duration')
    dura.text = str(REST_TYPE_TO_DURATION[rest.label])
    if rest.has_dot:
        du = int(dura.text)
        dura.text = str(round(du*1.5))
        elem.append(Element('dot'))
    staff = SubElement(elem, 'staff')
    staff.text = str(rest.track+1)
    return elem


def decode_backup(dura):
    if dura == 0:
        return None
    backup = Element('backup')
    du = SubElement(backup, 'duration')
    du.text = str(dura)
    return backup


def decode_forward(dura):
    if dura == 0:
        return None
    forward = Element('forward')
    du = SubElement(forward, 'duration')
    du.text = str(dura)
    return forward


def decode_clef(clef) -> Element:
    elem = Element('attributes')
    cc = SubElement(elem, 'clef', attrib={'number': str(clef.track+1)})
    sign = SubElement(cc, 'sign')
    sign.text = clef.label.name[0]
    line = SubElement(cc, 'line')
    line.text = '2' if sign.text == 'G' else '4'
    return elem


def decode_key(key) -> Element:
    elem = Element('attributes')
    kk = SubElement(elem, 'key')
    fifths = SubElement(kk, 'fifths')
    fifths.text = str(key.value)
    return elem


def decode_measure(measure, key=None, key_change=False):
    elem = Element('measure', attrib={'number': str(measure.number)})

    if key_change:
        attribute = SubElement(elem, 'attributes')
        div = SubElement(attribute, 'divisions')
        div.text = str(DIVISION_PER_QUATER)
        k = SubElement(attribute, 'key')
        fifth = SubElement(k, 'fifths')
        fifth.text = str(key.value)
        mode = SubElement(k, 'mode')
        mode.text = 'major'
        staves = SubElement(attribute, 'staves')
        staves.text = str(get_total_track_nums())
        if measure.has_clef:
            clefs = [sym for sym in measure.symbols if isinstance(sym, Clef)]
            for clef in clefs:
                cc = decode_clef(clef)
                attribute.append(cc)
    return elem


def pretty_xml(elem):
    return minidom.parseString(ET.tostring(elem)).toprettyxml(indent='  ', encoding='UTF-8')


def build_part_list():
    parts = Element('part-list')
    part = SubElement(parts, 'score-part', attrib={'id': 'P1'})
    p_name = SubElement(part, 'part-name')
    p_name.text = "Piano"
    score_inst = SubElement(part, 'score-instrument', attrib={'id': 'P1-I1'})
    inst_name = SubElement(score_inst, 'instrument-name')
    inst_name.text = "Piano"
    inst_sound = SubElement(score_inst, 'instrument-sound')
    inst_sound.text = "keyboard.piano"
    midi_inst = SubElement(part, 'midi-instrument', attrib={'id': 'P1-I1'})
    midi_ch = SubElement(midi_inst, 'midi-channel')
    midi_ch.text = '1'
    program = SubElement(midi_inst, 'midi-program')
    program.text = '1'
    volume = SubElement(midi_inst, 'volume')
    volume.text = '80'
    pan = SubElement(midi_inst, 'pan')
    pan.text = '0'

    return parts


def build_work(f_name=None):
    work = Element("work")
    title = SubElement(work, "work-title")
    title.text = f_name if f_name is not None else "Meteo AI Research"
    return work


def build_identity():
    iden = Element("identification")
    creator = SubElement(iden, "creator", attrib={"type": "composer"})
    creator.text = "by Meteo"
    return iden


if __name__ == "__main__":
    notes = layers.get_layer('notes')
    groups = layers.get_layer('note_groups')
    barlines = layers.get_layer('barlines')
    rests = layers.get_layer('rests')
    clefs = layers.get_layer('clefs')
    sfns = layers.get_layer('sfns')

    voices = get_voices()
    group_container = sort_symbols(voices)

    builder = MusicXMLBuilder()
    builder.build()
    xml = builder.to_musicxml()
    open("initial.musicxml", "wb").write(xml)

    nn = NoteHead()
    nn.staff_line_pos = -2
    nn.has_dot = True
    nn.stem_up = True
    nn.label = NoteType.QUARTER
    nn.track = 0
    nn.group = 0
    ee = pretty_xml(decode_note(nn, ClefType.G_CLEF)).decode('utf-8')
    #print(ee)

    rr = Rest()
    rr.label = RestType.EIGHTH
    rr.track = 0
    rr.group = 0
    ll = pretty_xml(decode_rest(rr)).decode('utf-8')
    #print(ll)

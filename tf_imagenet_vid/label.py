from collections import defaultdict
import os

from attrdict import AttrDict
import xml.etree.cElementTree as ElementTree

import numpy as np


def parse_object(obj):
    """Extracts object properties from an xml node representing an object."""
    id = int(obj[0].text)
    x2, x1, y2, y1 = (int(i.text) for i in obj[2])
    bbox = np.asarray([y1, x1, y2 - y1, x2 - x1])

    return id, dict(name=obj[1].text, bbox=bbox, occluded=int(obj[3].text),
                    generated=int(obj[4].text))


def parse_label(label_path):
    """Parses an xml label file."""
    root = ElementTree.parse(label_path).getroot()

    folder = root[0].text
    filename = root[1].text
    size = tuple(int(i.text) for i in reversed(root[3]))

    obj_by_id = dict()
    objects = root[4:]
    for obj in objects:
        id, obj = parse_object(obj)
        obj_by_id[id] = obj

    return folder, filename, size, obj_by_id


def find_xmls(path):
    """Finds all xml files in a path."""
    xmls = sorted([p for p in os.listdir(path) if p.endswith('.xml')])
    return [os.path.join(path, xml) for xml in xmls]


def parse_label_seq(seq_path, fixed_size=None):
    """Parses xml labels for a given sequence and potentially adjusts bbox size.

    Args:
        seq_path: string, path to a folder with xmls for a single sequence.

    Returns:
        tuple of ({folder: file_list}, {obj_id: properties})
    """
    xmls = find_xmls(seq_path)

    obj_by_id = dict()
    imgs_by_folder = defaultdict(list)

    for p in xmls:
        folder, filename, size, by_id = parse_label(p)

        # figure out how to scale the bounding box to reflect new image size
        if fixed_size:
            size_ratio = np.asarray(fixed_size).astype(np.float32)
            size_ratio = np.tile(size_ratio / np.asarray(size), (2,))

            size = fixed_size
        else:
            size_ratio = 1.

        imgs_by_folder[folder].append(dict(filename=filename, size=size))

        for i, obj in by_id.iteritems():

            if i not in obj_by_id:
                obj_by_id[i] = AttrDict()
                obj_by_id[i].id = i
                obj_by_id[i].name = obj['name']
                obj_by_id[i]['boxes'] = []

            obj = AttrDict(obj)
            del obj['name']
            # scale the bounding box
            obj.bbox = obj.bbox * size_ratio
            obj.frame_num = int(filename)
            obj_by_id[i]['boxes'].append(obj)

    return imgs_by_folder, obj_by_id
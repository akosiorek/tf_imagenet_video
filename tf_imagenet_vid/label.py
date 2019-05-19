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

def load_UADETRAC_annotation(path,img_size=None,scaling=1):
    """
    for a given sequence, load images and bounding boxes info from XML file
    :param index: index of a specific sequence
    :return: record['boxes', 'gt_classes', 'gt_overlaps', 'flipped']

    from
    https://github.com/wkelongws/RDFCN_UADETRAC_AICITY/blob/master/UA_DETRAC.py
    """
    import xml.etree.ElementTree as ET
    import cv2

    classes = ['__background__',  # always index 0
               'vehicle']
    num_classes = len(classes)

    roi_recs = []
    tree = ET.parse(path)
    frames = tree.findall('frame')
    for ix, frame in enumerate(frames):
        density = frame.attrib['density']  # string '7'
        frame_num = frame.attrib['num']  # string '1'
        img_num = ''
        if len(frame_num) < 5:
            for iter_img_num in range(5 - len(frame_num)):
                img_num = '0' + img_num
        img_num = img_num + frame_num

        roi_rec = dict()
        # roi_rec['image'] = os.path.join(self.image_path_from_index(index),
        #                                 'img' + img_num + '.jpg')
        # print(roi_rec['image'] )
        # im_size = cv2.imread(roi_rec['image'],
        #                      cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION).shape
        # print roi_rec['image'], im_size
        roi_rec['height'] = float(img_size[0])/scaling
        roi_rec['width'] = float(img_size[1])/scaling

        target_list = frame.findall('target_list')
        if len(target_list) > 0:
            tl = target_list[0]
            targets = tl.findall('target')

            # if not self.config['use_diff']:
            #    non_diff_objs = [obj for obj in objs if int(obj.find('difficult').text) == 0]
            #    objs = non_diff_objs
            num_targets = len(targets)

            boxes = np.zeros((num_targets, 4), dtype=np.uint16)
            gt_classes = np.zeros((num_targets), dtype=np.int32)
            overlaps = np.zeros((num_targets, num_classes),
                                dtype=np.float32)
            ids = np.zeros(num_targets)

            class_to_index = dict(
                zip(classes, range(num_classes)))
            # Load object bounding boxes into a data frame.
            for ix, target in enumerate(targets):
                bbox = target.find('box').attrib
                # Make pixel indexes 0-based
                x1 = (float(bbox['left']) - 1)/scaling
                y1 = (float(bbox['top']) - 1)/scaling
                x2 = (x1 + float(bbox['width']))/scaling
                y2 = (y1 + float(bbox['height']))/scaling

                # get object id; the attribute 'id' starts counting at 1,
                # we want to start counting at 0
                ids[ix] = int(target.attrib['id']) - 1

                # x1 = float(bbox.find('xmin').text) - 1
                # y1 = float(bbox.find('ymin').text) - 1
                # x2 = float(bbox.find('xmax').text) - 1
                # y2 = float(bbox.find('ymax').text) - 1
                # cls = class_to_index[obj.find('name').text.lower().strip()]
                cls = class_to_index['vehicle']
                # boxes[ix, :] = [x1, y1, x2, y2]

                boxes[ix, :] = [float(bbox['top']) - 1, float(bbox['left']), float(bbox['height']), float(bbox['width'])]
                boxes[ix, 0] /= scaling
                boxes[ix, 1] /= scaling
                boxes[ix, 2] /= scaling
                boxes[ix, 3] /= scaling

                gt_classes[ix] = cls
                overlaps[ix, cls] = 1.0

                # FAFU: this seems redundant, could also call one level above
                # (i.e. after the for loop)
                roi_rec.update({'boxes': boxes,
                                'gt_classes': gt_classes,
                                'gt_overlaps': overlaps,
                                'max_classes': overlaps.argmax(axis=1),
                                'max_overlaps': overlaps.max(axis=1),
                                'flipped': False,
                                'ids': ids})

        roi_recs.append(roi_rec)
    return roi_recs
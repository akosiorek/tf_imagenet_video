import os.path as osp

from attrdict import AttrDict

import numpy as np
from tf_imagenet_vid.label import parse_label_seq
from tf_imagenet_vid.label import load_UADETRAC_annotation

def extract_files(filedict):
    """Extracts a sorted list of files from a {folder: file_list} dict for a single seq.
    """

    files = []
    dirnames = sorted(filedict.keys(), key=lambda x: int(x.split('_')[-1]))
    for dirname in dirnames:
        filelist = filedict[dirname]
        filelist = [d['filename'] for d in filelist]
        filelist = sorted(filelist, key=lambda x: int(x.split('.')[0]))
        for f in filelist:
            files.append(osp.join(dirname, f) + '.JPEG')

    return files


def extract_properties(filedict, properties, root_dir=None):
    """Extracts properties from an {obj_id: properties} dict and converts them to
    numpy arrays.
    """

    img_files = extract_files(filedict)
    if root_dir is not None:
        img_files = [osp.join(root_dir, f) for f in img_files]

    n_obj = len(properties)
    num_frames = len(img_files)

    obj_to_idx = dict(zip(properties.keys(), range(n_obj)))

    boxes = np.zeros([num_frames, n_obj, 4], dtype=np.float32)
    generated = np.zeros([num_frames, n_obj, 1], dtype=np.uint8)
    occluded = np.zeros([num_frames, n_obj, 1], dtype=np.uint8)
    presence = np.zeros([num_frames, n_obj, 1], dtype=np.uint8)

    for obj_id in properties.keys():
        obj_idx = obj_to_idx[obj_id]
        for box in properties[obj_id].boxes:
            frame_num = box.frame_num
            boxes[frame_num, obj_idx, :] = box.bbox
            generated[frame_num, obj_idx] = box.generated
            occluded[frame_num, obj_idx] = box.occluded
            presence[frame_num, obj_idx] = 1

    return AttrDict(boxes=boxes, generated=generated, occluded=occluded,
                    presence=presence, n_obj=n_obj, img_files=img_files)

def process_uadetrac(seq, seq_name=None, video_dir=None):
    """Extracts properties from an {obj_id: properties} dict and converts them to
    numpy arrays.
    """

    # get number of objects in entire scene:
    n_objects = 0
    for frame in seq:
        n_objects = int(max(max(frame['ids']), n_objects))
    num_frames = len(seq)


    # img_files = extract_files(filedict)
    # if root_dir is not None:
    #     img_files = [osp.join(root_dir, f) for f in img_files]



    boxes = np.zeros([num_frames, n_objects, 4], dtype=np.float32)
    generated = np.zeros([num_frames, n_objects, 1], dtype=np.uint8)
    occluded = np.zeros([num_frames, n_objects, 1], dtype=np.uint8)
    presence = np.zeros([num_frames, n_objects, 1], dtype=np.uint8)
    img_files = []

    classes_observed = {}

    for i_f, frame in enumerate(seq):

        img_file = osp.join(video_dir,'%s/img%05d.jpg' % (seq_name, i_f+1))
        img_files.append(img_file)

        for i_id, id in enumerate(frame['ids']):
            if  frame['gt_classes'][i_id] == 1:
                boxes[i_f,i_id,:] = frame['boxes'][i_id]
                presence[i_f,i_id,:] = 1
            else:
                classes_observed[frame['gt_classes'][i_id]] = 1

    assert len(img_files) == num_frames

    return AttrDict(boxes=boxes, generated=generated, occluded=occluded,
                    presence=presence, n_obj=n_objects, img_files=img_files)


def parse(sequence_list, video_root_dir=None,
          img_size=None, seq_filter=None):
    """
    :param sequence_list:
    :param video_root_dir:
    :param img_size:
    :param seq_filter:
    :return: a list with length being number of videos
             each list element is a dict
             'boxes' : [T,K,4]
             'generated' : [T,K,1] - probably whether annotation was interpolation or manual
             'img_files' : [T] - list of strings with img filenames
             'n_obj' : scalar - number of objects in video
             'occluded' : [T,K,1]
             'presence' : [T,K,1]
    """

    parsed_seqs = []
    for seq in sequence_list:
        parsed_seqs.append(parse_label_seq(seq, fixed_size=img_size))

    parsed_seqs = np.asarray(parsed_seqs)
    if seq_filter is not None:
        parsed_seqs = seq_filter(parsed_seqs)

    seq_properties = []
    for parsed_seq in parsed_seqs:
        properties = extract_properties(*parsed_seq, root_dir=video_root_dir)
        seq_properties.append(properties)

    return np.asarray(seq_properties)


def parse_uadetrac(sequence_list, annotation_dir=None, video_root_dir=None,
          img_size=None, seq_filter=None):
    """
    :param sequence_list:
    :param video_root_dir:
    :param img_size:
    :param seq_filter:
    :return: a list with length being number of videos
             each list element is a dict
             'boxes' : [T,K,4]
             'generated' : [T,K,1] - probably whether annotation was interpolation or manual
             'img_files' : [T] - list of strings with img filenames
             'n_obj' : scalar - number of objects in video
             'occluded' : [T,K,1]
             'presence' : [T,K,1]
    """

    parsed_seqs = []
    processed_seqs = []
    for seq in sequence_list:
        path =  osp.join(annotation_dir, seq)
        parsed_seq = load_UADETRAC_annotation(path,img_size)
        parsed_seqs.append(parsed_seq)
        processed_seq = process_uadetrac(parsed_seq, seq_name=seq[:-4], video_dir=video_root_dir)
        processed_seqs.append(processed_seq)

    # parsed_seqs = np.asarray(parsed_seqs)
    # if seq_filter is not None:
    #     parsed_seqs = seq_filter(parsed_seqs)

    return processed_seqs
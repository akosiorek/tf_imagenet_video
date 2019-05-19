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

def process_uadetrac(seq, seq_name=None, video_dir=None, offset=0):
    """Extracts properties from an {obj_id: properties} dict and converts them to
    numpy arrays.
    """

    num_frames = len(seq)

    # get number of objects in entire scene:
    # n_objects = 0
    # for frame in seq:
    #     n_objects = int(max(max(frame['ids'])+1, n_objects))

    # get number of objects in this (sub)sequence:
    dict_ids = {}
    for frame in seq:
        for id in frame['ids']:
            dict_ids[int(id)] = True
    list_ids = dict_ids.keys()
    n_objects = len(list_ids)
    list_ids.sort()


    # img_files = extract_files(filedict)
    # if root_dir is not None:
    #     img_files = [osp.join(root_dir, f) for f in img_files]



    boxes = np.zeros([num_frames, n_objects, 4], dtype=np.float32)
    generated = np.zeros([num_frames, n_objects, 1], dtype=np.uint8)
    occluded = np.zeros([num_frames, n_objects, 1], dtype=np.uint8)
    presence = np.zeros([num_frames, n_objects, 1], dtype=np.uint8)
    img_files = []

    classes_ignored = {}

    for i_f, frame in enumerate(seq):

        img_file = osp.join(video_dir,'%s/img%05d.jpg' % (seq_name, i_f+1+offset))
        img_files.append(img_file)

        for i_id, id in enumerate(frame['ids']):
            # assign new id in bijective way so that resulting list of ids
            # is range(0, n_objects) where every id is used
            # new_id = int(list_ids[int(id)])
            new_id = list_ids.index(int(id))
            # ensure class of object is one (should typically be the case)
            if  frame['gt_classes'][i_id] == 1:
                boxes[i_f,new_id,:] = frame['boxes'][i_id]
                presence[i_f,new_id,:] = 1
            else:
                # track discarded object classes
                classes_ignored[frame['gt_classes'][i_id]] = 1

    assert len(img_files) == num_frames

    print('classes_ignored')
    print(classes_ignored)

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
          img_size=None, seq_filter=None, split=True, scaling=1):
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
        parsed_seq = load_UADETRAC_annotation(path,img_size, scaling)
        # parsed_seqs.append(parsed_seq)
        if split:
            splitted, offsets = split_unprocessed(parsed_seq)
            for i_split, splitted_element in enumerate(splitted):
                processed_seq = process_uadetrac(splitted_element,
                                                 seq_name=seq[:-4],
                                                 video_dir=video_root_dir,
                                                 offset=offsets[i_split])
                processed_seqs.append(processed_seq)
        else:
            processed_seq = process_uadetrac(parsed_seq, seq_name=seq[:-4], video_dir=video_root_dir)
            processed_seqs.append(processed_seq)

    # parsed_seqs = np.asarray(parsed_seqs)
    # if seq_filter is not None:
    #     parsed_seqs = seq_filter(parsed_seqs)

    return processed_seqs


# FAFU 19/05/13: currently not needed, only using split_unprocessed(...)
def split_processed(parsed_seqs):
    splitted = []
    for i_seq in range(len(parsed_seqs)):
        len_i = len(parsed_seqs[i_seq]['img_files'])
        for t in range(len_i - 100):
            if t%50==0:
                t_dict = {}
                for key in parsed_seqs[i_seq]:
                    # copy all dict entries over except n_obj as this
                    # piece of information will change
                    if key != 'n_obj':
                        t_dict[key] = parsed_seqs[i_seq][key][t:t+100]
                splitted.append(t_dict)
    return splitted

def split_unprocessed(unparsed_seq):
    num_frames = len(unparsed_seq)
    offsets = []
    splitted = []
    for t in range(num_frames - 100):
        if t%50==0:
            t_seq = unparsed_seq[t:t+100]
            splitted.append(t_seq)
            offsets.append(t)
    return splitted, offsets
import os
import sys
import time

from tf_imagenet_vid.parse import parse
from tf_imagenet_vid.parse import parse_uadetrac
from tf_imagenet_vid.write_dataset import save_seqs_as_tfrecords

import argparse
from util import log_imgs_seqs

# command line argument parser
ARGPARSER = argparse.ArgumentParser(
    description='Load ImagenetVideo Dataset')
ARGPARSER.add_argument(
    '--video_path', type=str,
    default='/Users/adam/data/ILSVRC2015_VID/Data/VID/train',
    help='path to  .gzip file')
ARGPARSER.add_argument(
    '--annotations_path', type=str,
    default='/Users/fabian/Documents/Code/Tracking/tf_imagenet_video/ILSVRC2015_VID/Annotations/VID/train',
    help='path to  .gzip file')
ARGPARSER.add_argument(
    '--seq_folder', type=str,
    default='ILSVRC2015_VID_train_0000',
    help='path to  .gzip file')
# ARGPARSER.add_argument(
#     '--video_path', type=str,
#     default='/Users/adam/data/ILSVRC2015_VID/Data/VID/train',
#     help='path to  .gzip file')
# /Users/fabian/Documents/Code/Tracking/tf_imagenet_video/ILSVRC2015_VID/Data/VID/train/ILSVRC2015_VID_train_0000

FLAGS, UNPARSED_ARGV = ARGPARSER.parse_known_args()



VIDEO_ROOT_DIR = FLAGS.video_path


def find_sequences(seq_folder, mode='folders'):

    # mode describes in which way sequences are stored
    # 'folders': one folder with many xml files per sequence
    # 'files': one xml file per sequence

    seqs = []
    for root, dirs, files in os.walk(seq_folder):
        if mode=='folders':
            if files:
                if '.xml' in files[0]:
                    seqs.append(root)
                    continue
        elif mode=='files':
            for file in files:
                if '.xml' in file:
                    seqs.append(file)

    return seqs


if __name__ == '__main__':
    """
    :param seq_folder: path to top-level folder containing folders
        with annotations.
    """

    # sequence_list is a list of strings of annotation folders with each
    # folder containing frames of a video
    # len(sequence_list) is number of videos being loaded
    VIDEO_ROOT_DIR = '/Users/fabian/Documents/Code/Tracking/Data/Detrac/Train_imgs'
    train = 1

    if train:
        ANNOTATION_DIR = '/Users/fabian/Documents/Code/Tracking/Data/Detrac/Split_train_XML'
        tfrecords_file = 'split_train.tfrecords'
    else:

        ANNOTATION_DIR = '/Users/fabian/Documents/Code/Tracking/Data/Detrac/Split_eval_XML'
        tfrecords_file = 'split_eval.tfrecords'


    # tested a few images by loading them with cv2.imread(path).shape, all had
    # shape (540, 960, 3)
    IMG_SIZE = (540, 960)

    sequence_list = find_sequences(ANNOTATION_DIR, mode='files')
    # sequence_list_fullpaths_annot = [os.path.join(seq_folder, i) for i in sequence_list]

    # parsed_seqs is a list with length being number of videos
    # each list element is a dict
    # 'boxes' : [T,K,4]
    # 'generated' : [T,K,1] - probably whether annotation was interpolation or manual
    # 'img_files' : [T] - list of strings with img filenames
    # 'n_obj' : scalar - number of objects in video
    # 'occluded' : [T,K,1]
    # 'presence' : [T,K,1]
    time_c = time.time()
    parsed_seqs = parse_uadetrac(sequence_list, ANNOTATION_DIR, VIDEO_ROOT_DIR, img_size=IMG_SIZE)
    print('%.2fs' % (time.time() - time_c))

    log_imgs = 1
    if log_imgs:
        pred = {'inpt': [parsed_seqs[0]['img_files']],
                'visibility': [parsed_seqs[0]['presence']],
                'coords': [parsed_seqs[0]['boxes']]}
        log_imgs_seqs(pred, max_frames=50, inpts_are_filenames=1, max_obj=5)

    time_c = time.time()
    save_seqs_as_tfrecords(parsed_seqs, tfrecords_file, img_size=IMG_SIZE)
    print('%.2fs' % (time.time() - time_c))

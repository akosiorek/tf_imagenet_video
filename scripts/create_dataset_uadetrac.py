import os
import sys
import time

from tf_imagenet_vid.parse import parse
from tf_imagenet_vid.parse import parse_uadetrac
from tf_imagenet_vid.write_dataset import save_seqs_as_tfrecords
from tf_imagenet_vid.write_dataset import save_seqs_as_gzip

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

    if sys.platform == "darwin":
        user = '/Users'
    else:
        user = '/home'

    # sequence_list is a list of strings of annotation folders with each
    # folder containing frames of a video
    # len(sequence_list) is number of videos being loaded
    VIDEO_ROOT_DIR = user+'/fabian/Documents/Code/Tracking/Data/Detrac/Train_imgs'
    train = 0

    if train:
        ANNOTATION_DIR = user+'/fabian/Documents/Code/Tracking/Data/Detrac/Split_train_XML'
        output_file = 'split_train.tfrecords'
    else:
        ANNOTATION_DIR = user+'/fabian/Documents/Code/Tracking/Data/Detrac/Split_eval_XML'
        output_file = 'split_test.tfrecords'

        ANNOTATION_DIR = user+'/fabian/Documents/Code/Tracking/Data/Detrac/Split_eval_XML_small'
        output_file = 'split_test_small.gzip'

    split = not ('.gzip' in output_file)

    # tested a few images by loading them with cv2.imread(path).shape, all had
    # shape (540, 960, 3)
    IMG_SIZE = [540, 960]
    SCALING = 2.5
    IMG_SIZE[0] = int(IMG_SIZE[0]/SCALING)
    IMG_SIZE[1] = int(IMG_SIZE[1]/SCALING)

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
    parsed_seqs = parse_uadetrac(sequence_list, ANNOTATION_DIR, VIDEO_ROOT_DIR, img_size=IMG_SIZE, scaling=SCALING, split=split)
    print('%.2fs' % (time.time() - time_c))

    # split sequences into subsequences of length 100
    # this has multiple reasons:
    # - number of batches per epoch increase (better queuing)
    # - loading of smaller sequences may be faster


    # plot one sequence to folder 'deleteMe'
    log_imgs = 0
    if log_imgs:
        pred = {'inpt': [parsed_seqs[3]['img_files']],
                'visibility': [parsed_seqs[3]['presence']],
                'coords': [parsed_seqs[3]['boxes']]}
        log_imgs_seqs(pred, max_frames=50, inpts_are_filenames=1, max_obj=5, img_size=IMG_SIZE)

    time_c = time.time()
    if '.tfrecords' in output_file:
        save_seqs_as_tfrecords(parsed_seqs, output_file, img_size=IMG_SIZE)
    elif '.gzip' in output_file:
        save_seqs_as_gzip(parsed_seqs, output_file, img_size=IMG_SIZE)
    else:
        print('not a valid file extension')
        exit()

    print('%.2fs' % (time.time() - time_c))

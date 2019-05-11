import os
import sys
import time

from tf_imagenet_vid.parse import parse
from tf_imagenet_vid.write_dataset import save_seqs_as_tfrecords

import argparse
from util import log_imgs_one_seq

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


IMG_SIZE = (480, 640)
VIDEO_ROOT_DIR = FLAGS.video_path


def find_sequences(seq_folder):

    seqs = []
    for root, dirs, files in os.walk(seq_folder):
        if files:
            if '.xml' in files[0]:
                seqs.append(root)
                continue

    return seqs


def main(seq_folder, tfrecords_file):
    """
    :param seq_folder: path to top-level folder containing folders
        with annotations.
    """


    # sequence_list is a list of strings of annotation folders with each
    # folder containing frames of a video
    # len(sequence_list) is number of videos being loaded
    time_c = time.time()
    sequence_list = find_sequences(seq_folder)
    print('%.2fs' %(time.time() - time_c))

    # parsed_seqs is a list with length being number of videos
    # each list element is a dict
    # 'boxes' : [T,K,4]
    # 'generated' : [T,K,1] - probably whether annotation was interpolation or manual
    # 'img_files' : [T] - list of strings with img filenames
    # 'n_obj' : scalar - number of objects in video
    # 'occluded' : [T,K,1]
    # 'presence' : [T,K,1]
    time_c = time.time()
    parsed_seqs = parse(sequence_list, VIDEO_ROOT_DIR, img_size=IMG_SIZE)
    print('%.2fs' % (time.time() - time_c))

    time_c = time.time()
    parsed_n5 = []
    count_samples = 0
    for seq_el in parsed_seqs:
        if seq_el['n_obj'] == 5:
            parsed_n5.append(seq_el)
            count_samples += 1
            if count_samples >= 1:
                break
    print('%.2fs' % (time.time() - time_c))

    time_c = time.time()
    save_seqs_as_tfrecords(parsed_n5, tfrecords_file, img_size=IMG_SIZE)
    print('%.2fs' % (time.time() - time_c))

if __name__ == '__main__':
    # main(*sys.argv[1:])
    main(seq_folder = os.path.join(FLAGS.annotations_path,FLAGS.seq_folder), tfrecords_file='tfrecordsfile')


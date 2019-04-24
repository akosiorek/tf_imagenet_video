import os
import sys

from tf_imagenet_vid.parse import parse
from tf_imagenet_vid.write_dataset import save_seqs_as_tfrecords


IMG_SIZE = (480, 640)
VIDEO_ROOT_DIR = '/Users/adam/data/ILSVRC2015_VID/Data/VID/train'


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
    sequence_list = find_sequences(seq_folder)[:10]
    parsed_seqs = parse(sequence_list, VIDEO_ROOT_DIR, img_size=IMG_SIZE)
    save_seqs_as_tfrecords(parsed_seqs, tfrecords_file, img_size=IMG_SIZE)

if __name__ == '__main__':
    main(*sys.argv[1:])


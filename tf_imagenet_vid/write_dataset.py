import io

import numpy as np
import tensorflow as tf

from PIL import Image


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_feature_list(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def read_img_as_bytes(path, img_size=None):
    buffer = io.BytesIO()
    img = Image.open(path)

    if img_size is not None:
        img = img.resize(reversed(img_size))

    img.save(buffer, format='JPEG')
    return buffer.getvalue()


def video_example(properties, img_size=None):
    img_files = list(properties.img_files)

    img_shape = np.asarray(Image.open(img_files[0])).shape
    if img_size is None:
        img_string = [open(f, 'rb').read() for f in img_files]

    else:
        img_string = [read_img_as_bytes(f, img_size) for f in img_files]
        img_shape = list(img_size) + [img_shape[-1]]

    feature = {
        'height': _int64_feature(img_shape[0]),
        'width': _int64_feature(img_shape[1]),
        'depth': _int64_feature(img_shape[2]),
        'num_frames': _int64_feature(len(img_files)),
        'frames_raw': _bytes_feature_list(img_string),
        'n_obj': _int64_feature(properties.n_obj)
    }

    for k in 'boxes generated occluded presence'.split():
        feature[k] = _bytes_feature_list(properties[k].tobytes())

    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_seqs_as_tfrecords(seq_list, tfrecords_file, img_size=None):
    with tf.python_io.TFRecordWriter(tfrecords_file) as writer:
        for seq_props in seq_list:
            vid = video_example(seq_props, img_size)
            vid_string = vid.SerializeToString()
            writer.write(vid_string)

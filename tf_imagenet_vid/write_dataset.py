import io
import gzip
import pickle

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
        # this is using the PIL.Image.Imgae.resize function
        # (not numpy.resize!), documentation:
        # https://pillow.readthedocs.io/en/3.1.x/reference/Image.html
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

    # we want to create an instance of tf.train.Features which is a collection
    # of named features.
    # It has a single attribute feature (defined in the next line)
    # that expects a dictionary where the key is the name of the features
    # and the value a tf.train.Feature.
    # (beware of the confusing naming convention: feature is a dictionary
    # of features, i.e. each entry is a feature)
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

    # tf.train.Example is one of the main components for structuring a TFRecord.
    # An tf.train.Example stores features in a single attribute features of
    # type tf.train.Features.
    return tf.train.Example(features=tf.train.Features(feature=feature))


def save_seqs_as_tfrecords(seq_list, tfrecords_file, img_size=None):

    # tf.python_io.TFRecordWriter (in contrast to tf.train.Features etc.)
    # is actually a Python class. It accepts a file path in its path attribute
    # and creates a writer object that works just like any other file object.
    # The TFRecordWriter class offers write, flush and close methods.
    # The method write accepts a string as parameter and writes it to disk,
    # meaning that structured data must be serialized first. To this end,
    # tf.train.Example and tf.train.SequenceExample provide SerializeToString
    # methods:
    with tf.python_io.TFRecordWriter(tfrecords_file) as writer:
        for i, seq_props in enumerate(seq_list):
            vid = video_example(seq_props, img_size)
            vid_string = vid.SerializeToString()
            writer.write(vid_string)
            print('written ' + str(i+1) + ' of ' + str(len(seq_list)))
    # It is important that the type of a feature is the same across all samples
    # in the dataset


def save_seqs_as_gzip(seq_list, gzip_file, img_size=None):
    data_mode = {'inpt': [],'coords': [],'presence': [],'visibility': []}
    for i, seq_props in enumerate(seq_list):
        data_mode['coords'].append(seq_props['boxes'])
        data_mode['presence'].append(seq_props['presence'])
        data_mode['visibility'].append(seq_props['presence'])

        data_mode['inpt'].append([])
        for img_file in seq_props['img_files']:
            img = Image.open(img_file)
            img = img.resize(img_size[::-1])
            img = np.asarray(img)
            data_mode['inpt'][i].append(img)
        data_mode['inpt'][i] = np.asarray(data_mode['inpt'][i])
        print('loaded imgs for seq. ' + str(i+1) + ' of ' + str(len(seq_list)))
    data_mode['inpt'] = np.asarray(data_mode['inpt'])
    data_mode['coords'] = np.asarray(data_mode['coords'])
    data_mode['presence'] = np.asarray(data_mode['presence'])
    data_mode['visibility'] = np.asarray(data_mode['visibility'])
    with gzip.open(gzip_file, 'wb', 4) as f:
        pickle.dump(data_mode, f, pickle.HIGHEST_PROTOCOL)

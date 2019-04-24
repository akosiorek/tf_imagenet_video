import tensorflow as tf
from tensorflow.python.util import nest


class Dataset(object):
    _feature_description = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'depth': tf.FixedLenFeature([], tf.int64),
        'num_frames': tf.FixedLenFeature([], tf.int64),
        'frames_raw': tf.VarLenFeature(tf.string),
        'n_obj': tf.FixedLenFeature([], tf.int64),
        'boxes': tf.VarLenFeature(tf.string),
        'generated': tf.VarLenFeature(tf.string),
        'occluded': tf.VarLenFeature(tf.string),
        'presence': tf.VarLenFeature(tf.string),
    }

    def __init__(self, tfrecord_files, img_shape, max_n_objects=None):
        super(Dataset, self).__init__()
        self._tfrecords = nest.flatten(tfrecord_files)
        self._img_shape = img_shape
        self._max_n_objects = max_n_objects

    def __call__(self):
        ds = tf.data.TFRecordDataset(self._tfrecords)
        return ds.map(self._parse_example)

    def _parse_example(self, example_proto):
        d = tf.parse_single_example(example_proto, self._feature_description)

        # # play with the below to create random subsequences.
        # #
        # def map_func(idx):
        #     return tf.io.decode_jpeg(d['frames_raw'].values[idx])
        #
        # random_offset = tf.random_uniform(
        #     shape=(), minval=0,
        #     maxval=d['num_frames'] - SEQ_NUM_FRAMES, dtype=tf.int64)
        #
        # offsets = tf.range(random_offset, random_offset + SEQ_NUM_FRAMES,
        #                    EVERY_N_FRAMES)
        # img = tf.map_fn(map_func, offsets, dtype=tf.uint8)

        img = tf.map_fn(tf.io.decode_jpeg, d['frames_raw'].values,
                        dtype=tf.uint8)

        img.set_shape([None] + list(self._img_shape))
        d['frames'] = img
        del d['frames_raw']

        n_obj = d['n_obj']
        num_frames = d['num_frames']
        _shape = [num_frames, n_obj]

        # different sequences will have different number of objects;
        # here we make sure that they are padded accordingly
        if self._max_n_objects:
            n_pads = self._max_n_objects - n_obj

        def _decode(name, dtype, n_elems):
            encoded = d[name].values

            # tensor consists of strings of length 1, but we need 4 bytes
            #  for a float32 we need to merge 4 strings into one along the
            #  last axis
            if dtype == tf.float32:
                encoded = tf.reshape(encoded, [-1, 4])
                encoded = tf.unstack(encoded, axis=-1)
                encoded = tf.strings.join(encoded)

            decoded = tf.reshape(tf.decode_raw(encoded, dtype),
                                 _shape + [n_elems])

            if self._max_n_objects:
                decoded = tf.pad(decoded, [(0, 0), (0, n_pads), (0, 0)])

            d[name] = decoded

        _decode('occluded', tf.uint8, 1)
        _decode('generated', tf.uint8, 1)
        _decode('presence', tf.uint8, 1)
        _decode('boxes', tf.float32, 4)

        return d
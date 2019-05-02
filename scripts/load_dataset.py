import os
import sys
import tensorflow as tf
from scipy.misc import imshow
tf.enable_eager_execution()
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from attention.model import util
from matplotlib import patches
import numpy as np


# load .tfrecords file
# this would be my training data; when done with one epoch, reset graph and
# do this again for loading validation data
# raw_image_dataset = tf.data.TFRecordDataset('10images.tfrecords')
raw_image_dataset = tf.data.TFRecordDataset('tfrecordsfile_5obj_1scene')


# Create a dictionary describing the features.
image_feature_description = {
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

SEQ_NUM_FRAMES = 10
# EVERY_N_FRAMES = 2
# shape = (SEQ_NUM_FRAMES / EVERY_N_FRAMES, 360, 540, 3)

IMG_SIZE = (480, 640)
shape = (None, IMG_SIZE[0], IMG_SIZE[1], 3)
# for padding: has to be maximum number objects ever present in a sequence
# (not necessarily at the same time). if chosen too large, performance loss
# if chosen too small, will get an error when padding with negative number
MAX_N_OBJECTS = 30
# how many objects to track simultaneously
N_OBJECTS = 4



def log_imgs_from_pred(pred, log_prefix='train', log_dir='deleteMe'):
    # pred is dict with entries: 'images', 'presence',
    #                            'gt_bbox',

    for n in xrange(len(pred['inpt'])):
        name = '{}_seq_{}'.format(log_prefix, n)
        im_dir = os.path.join(log_dir, log_prefix + '_log', name)
        util.try_mkdir(im_dir)
        # outputs = sess.run(
        #     [x, y, p, model.pred_bbox, model.att_pred_bbox, model.glimpse],
        #     feed_dict)
        # # image, ground truth bb, presence, pred. bb, attention bb, glimpse
        # i, b, pres, pb, ab, g = [c[:, n] for c in outputs]

        im = pred['inpt'][n]
        b = pred['coords'][n]
        vis = pred['visibility'][n]

        # at this point all the variables only have one sample (essentially
        # batch_size == 1)
        # T(-> loop over t) is number of images, i.e. number of time steps
        T = len(im)
        for t in range(T):

            fig, ax = plt.subplots(1)

            im_cur = np.array(im[t]).squeeze()

            ax.imshow(im_cur, cmap=matplotlib.cm.Greys_r)

            # loop over number of objects
            for i in range(np.array(b).shape[1]):
                # Create a Rectangle patch
                # image_format
                # (expects x, y, w, h with x, y describing the left-top corner)
                # TODO: in the online doc it says bottom left corner
                rect = patches.Rectangle((b[t][i][1], b[t][i][0]),
                                         b[t][i][3],
                                         b[t][i][2], linewidth=1,
                                         edgecolor='r',
                                         facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)


                # presence values
                # these are matplotlib.patch.Patch properties
                props = dict(boxstyle='round', alpha=0.5)
                # image_format
                # place a text box in upper left in axes coords
                # transform = ax.transAxes,
                # i is object number
                props['facecolor'] = 'red'
                ax.text(b[t][i][1] + b[t][i][3] / 2.0 - 3,
                        b[t][i][0] + b[t][i][2] / 2.0 - 8,
                        '%.2f' % (vis[t][i]),
                        fontsize=5,
                        verticalalignment='top', bbox=props)

            # Save it
            filename = im_dir + '/' + str(t) + '.png'
            plt.savefig(filename)
            plt.close('all')


# example_proto is a part of a tf record (i.e. one video / image sequence?)
def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  d = tf.parse_single_example(example_proto, image_feature_description)

  #############################################################################
  # ADAMS VERSION OF CUTTING OUT SUBSEQUENCES
  # get scalar value with a valid starting point for current sequence
  random_offset = tf.random_uniform(shape=(), minval=0,
      maxval=d['num_frames'] - SEQ_NUM_FRAMES, dtype=tf.int64)

  random_offset = 290

  def map_func(idx):
      return tf.io.decode_jpeg(d['frames_raw'].values[idx])
  # additional, third argument for the tf.range function could specify how many
  # frames are skipped in each time step
  offsets = tf.range(random_offset, random_offset + SEQ_NUM_FRAMES)
  img = tf.map_fn(map_func, offsets, dtype=tf.uint8)
  #############################################################################

  # the following is the alternative decoding without cutting sequence length
  # decode jpeg-bytestrings to tensorflow arrays (?)
  # img = tf.map_fn(tf.io.decode_jpeg, d['frames_raw'].values, dtype=tf.uint8)

  # set_shape does not change the shape, it just adds shape information
  img.set_shape(shape)
  d['frames'] = img

  # delete the jpeg bytestring as not used anymore
  del d['frames_raw']

  # number of objects in current sequence
  n_obj = d['n_obj']
  # number of frames in current sequence
  num_frames = d['num_frames']
  _shape = [num_frames, n_obj]

  # different sequences will have different number of objects;
  # here we make sure that they are padded accordingly
  n_pads = MAX_N_OBJECTS - n_obj

  def _decode(name, dtype, n_elems):
    encoded = d[name].values

    # tensor consists of strings of length 1, but we need 4 bytes for a float32
    # we need to merge 4 strings into one along the last axis
    if dtype == tf.float32:
      encoded = tf.reshape(encoded, [-1, 4])
      encoded = tf.unstack(encoded, axis=-1)
      encoded = tf.strings.join(encoded)

    decoded = tf.reshape(tf.decode_raw(encoded, dtype), _shape + [n_elems])
    decoded = tf.pad(decoded, [(0, 0), (0, n_pads), (0, 0)])
    # case no cutting of sequence length
    # d[name] = decoded
    # case cutting of sequence length
    d[name] = decoded[random_offset:random_offset + SEQ_NUM_FRAMES]

  _decode('occluded', tf.uint8, 1)
  _decode('generated', tf.uint8, 1)
  _decode('presence', tf.uint8, 1)
  _decode('boxes', tf.float32, 4)

  #############################################################################
  # RANDOMLY SAMPLE N_OBJECTS
  # ... out of all the objects which are present in first frame
  # valid is a tenser which tells us which objects are present in first frame
  valid = d['presence'][0]
  valid = tf.reshape(valid, (MAX_N_OBJECTS,))
  # number of present objects
  n_present = tf.reduce_sum(tf.cast(valid, tf.float32))
  valid_idx = tf.boolean_mask(tf.range(MAX_N_OBJECTS), valid)
  # Shuffled valid indices
  valid_idx_shuffled = tf.random.shuffle(valid_idx)
  # Pick sample from shuffled indices
  # handles cases N_OBJECTS > len(valid_idx_shuffled) automatically:
  # len(valid_idx_sample) will be min(N_OBJECTS, len(valid_idx_shuffled))
  valid_idx_sample = valid_idx_shuffled[:N_OBJECTS]

  d['presence'] = tf.gather(d['presence'], valid_idx_sample, axis=1)
  d['generated'] = tf.gather(d['generated'], valid_idx_sample, axis=1)
  d['occluded'] = tf.gather(d['occluded'], valid_idx_sample, axis=1)
  d['boxes'] = tf.gather(d['boxes'], valid_idx_sample, axis=1)

  # if less than N_OBJECTS objects were present, padding is necesarry
  pad_final = N_OBJECTS - tf.minimum(n_present, N_OBJECTS)
  d['presence'] = tf.pad(d['presence'], [(0, 0), (0, pad_final), (0, 0)])
  d['generated'] = tf.pad(d['generated'], [(0, 0), (0, pad_final), (0, 0)])
  d['occluded'] = tf.pad(d['occluded'], [(0, 0), (0, pad_final), (0, 0)])
  d['boxes'] = tf.pad(d['boxes'], [(0, 0), (0, pad_final), (0, 0)])
  #############################################################################


  return d

# map maps '_parse_image_function' across all elements of 'raw_image_dataset'
# however, it does not do this here but instead 'on demand', so can be queued
# and parallised
parsed_image_dataset = raw_image_dataset.map(_parse_image_function, num_parallel_calls=4)
# call once per epoch
data_iterator = parsed_image_dataset.make_one_shot_iterator()
# use this directly to build graph on
# when gone through entire dataset, will throw error
# will probably be dictionary
fetched_data = data_iterator.get_next()

pred = {'inpt': [fetched_data['frames'].numpy()],
        'coords': [fetched_data['boxes'].numpy()],
        'visibility': [fetched_data['presence'].numpy()],
        }
log_imgs_from_pred(pred)


# parsed_image_dataset is an iterator (or can be treated as one?); here we are
# looping over it and could print out all the images etc. if we wanted
# .take(3) says that only the first 3 examples are looped over
# for image_features in parsed_image_dataset.take(3):
#   image_sequence = image_features['frames'].numpy()
#   # stop after getting first sequence
#
#   pred = {'inpt':[image_sequence],
#           'coords': [image_features['boxes'].numpy()],
#           'visibility': [image_features['presence'].numpy()],
#   }
#   log_imgs_from_pred(pred)
#
#   break

# print out first image of current sequence
# imshow(image_sequence[0])


print('finished')

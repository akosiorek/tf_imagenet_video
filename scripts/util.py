import os
import sys
from scipy.misc import imshow
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from attention.model import util
from matplotlib import patches
import numpy as np
from PIL import Image


def log_imgs_seqs(pred, log_prefix='train', log_dir='deleteMe', max_frames=None,
                  max_obj=None, inpts_are_filenames=False):
  # logs one sequence
  # pred is dict with entries:
  #   'inpt':   [T, H, W, 3]
  #   'visibility': [T, K, 1]
  #   'coords':  [T, K, 4]

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
    T = -max(-len(im), -max_frames)
    for t in range(T):

      fig, ax = plt.subplots(1)

      if inpts_are_filenames:
        im_cur = np.asarray(Image.open(im[t]))
      else:
        im_cur = np.array(im[t]).squeeze()

      ax.imshow(im_cur, cmap=matplotlib.cm.Greys_r)

      # loop over number of objects
      K = -max(-np.array(b).shape[1], -max_obj)
      for i in range(K):
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

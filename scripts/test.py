import tensorflow as tf
tf.enable_eager_execution()

def sample_indices(valid, m, seed=None):
    valid = tf.convert_to_tensor(valid)
    n = tf.size(valid)
    # Flatten boolean tensor
    valid_flat = tf.reshape(valid, [n])
    # Get flat indices where the tensor is true
    valid_idx = tf.boolean_mask(tf.range(n), valid_flat)
    # Shuffled valid indices
    valid_idx_shuffled = tf.random.shuffle(valid_idx, seed=seed)
    # Pick sample from shuffled indices
    valid_idx_sample = valid_idx_shuffled[:m]
    # Unravel indices
    return tf.transpose(tf.unravel_index(valid_idx_sample, tf.shape(valid)))

# with tf.Graph().as_default(), tf.Session() as sess:
valid = [[ True,  True, False,  True],
         [False,  True,  True, False],
         [False,  True, False, False]]
m = 4
sample_indices(valid, m, seed=0)
    # print(sess.run(sample_indices(valid, m, seed=0)))
    # [[1 1]
    #  [1 2]
    #  [0 1]
    #  [2 1]]
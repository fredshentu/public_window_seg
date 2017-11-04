import tensorflow as tf
import numpy as np


scales = tf.convert_to_tensor([2**(-0.25),1.0, 2**(0.25), 2**(0.5), 2**0.75, 2**(-0.5), 2**(-0.75), \
                                            2**(-1.0),2**(1.0)])






def score_sampling(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'data': tf.FixedLenFeature([np.prod(obs_shape)], tf.float32),
                  'center_max_axis':tf.FixedLenFeature([3], tf.float32),
          })
    center_max_axis = tf.cast(features['center_max_axis'], tf.float32)
    center_max_axis = tf.reshape(center_max_axis, [3])
    data = tf.cast(features['data'], tf.float32)
    data = tf.reshape(data, obs_shape)
    background = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    
    center_x = center_max_axis[0]
    center_y = center_max_axis[1]
    long_axis = center_max_axis[2]

    scale = tf.pow()
import tensorflow as tf
import numpy as np


scales = tf.convert_to_tensor([2**(-1), 2**(-0.75), 2**(-0.5), 2**0.5, 2**(0.75), 2])


inpSize = 192
maskSize = 112

def crop_data(data, xc, yc, side):
    data = tf.image.crop_to_bounding_box(data, tf.cast(yc-side/2.0, tf.int32), tf.cast(xc-side/2.0, tf.int32), tf.cast(side, tf.int32) , tf.cast(side, tf.int32))
    data = tf.image.random_flip_up_down(data)
    data = tf.image.random_flip_left_right(data)
    image = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [3]), [1,2,0])

    image = tf.image.resize_images(image, [inpSize, inpSize])

    # import pdb; pdb.set_trace()
    mask = tf.image.resize_images(mask, [maskSize, maskSize])
    mask = (mask + 1) / 2
    mask = tf.cast(tf.transpose(mask, [2,0,1])[0], tf.int32)

    return image, mask



def read_decode_positive_example_poking(filename_queue, shift=24):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'data': tf.FixedLenFeature([np.prod([240,240,4])], tf.float32),
                  'center_max_axis':tf.FixedLenFeature([3], tf.float32),
          })
    center_max_axis = tf.cast(features['center_max_axis'], tf.float32)
    center_max_axis = tf.reshape(center_max_axis, [3])
    maxDim = center_max_axis[2]/1.75

    data = tf.cast(features['data'], tf.float32)
    data = tf.reshape(data, [240,240,4])
    data = tf.pad(data,[[200,200],[200,200],[0,0]])
    background = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    
    
    scale = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    scale = scale * maxDim * 224.0/(160.0*128.0)
    side = scale * inpSize

    xc = center_max_axis[0] + 200
    yc = center_max_axis[1] + 200

    xc += tf.cast(tf.random_uniform([1], minval=-shift, maxval=shift, dtype=tf.int32)[0], tf.float32) * scale
    yc += tf.cast(tf.random_uniform([1], minval=-shift, maxval=shift, dtype=tf.int32)[0], tf.float32) * scale

    image, mask = crop_data(data, xc, yc, side)

    return image, mask, 1, background



def read_decode_negative_example_poking(filename_queue, neg_shift_min=46, neg_shift_max=64, pad_size=200):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'data': tf.FixedLenFeature([np.prod([240,240,4])], tf.float32),
                  'center_max_axis':tf.FixedLenFeature([3], tf.float32),
          })
    center_max_axis = tf.cast(features['center_max_axis'], tf.float32)
    center_max_axis = tf.reshape(center_max_axis, [3])
    maxDim = center_max_axis[2]/1.75

    data = tf.cast(features['data'], tf.float32)
    data = tf.reshape(data, [240,240,4])
    data = tf.pad(data,[[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    background = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])


    #if good range, using bad shift, if bad range, using arbitrary shift
    random_index = tf.random_uniform([1], minval=0, maxval=5, dtype=tf.int32)[0]
    

    scale1 = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    scale1 = scale1
    scale2 = scales[random_index]

    coin_flip = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.int32)[0]
    
    scale = tf.where(coin_flip < 1, x = scale1, y = scale2)
    scale = scale * maxDim * 224.0/(160.0*128.0)
    side = scale * inpSize

    shift_min = tf.where(coin_flip < 1, x = neg_shift_min, y = 0)
    shift_max = neg_shift_max

    xc = center_max_axis[0] + pad_size
    yc = center_max_axis[1] + pad_size

    xc += tf.cast(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.int32)[0]*2-1, tf.float32) * tf.cast(tf.random_uniform([1], minval=shift_min, maxval=shift_max, dtype=tf.int32)[0], tf.float32) * scale
    yc += tf.cast(tf.random_uniform([1], minval=0, maxval=1, dtype=tf.int32)[0]*2-1, tf.float32) * tf.cast(tf.random_uniform([1], minval=shift_min, maxval=shift_max, dtype=tf.int32)[0], tf.float32) * scale

    image, mask = crop_data(data, xc, yc, side)

    return image, mask, 0, background


def inputs_poking(filenames, pos_max=24, neg_min=46, neg_max=64, batch_size=16, positive=True, viz=False):
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        if positive:
            image, mask, score, background = read_decode_positive_example_poking(filename_queue, \
                                            shift=pos_max)
        
        else:
            image, mask, score, background = read_decode_negative_example_poking(filename_queue, \
                                            neg_shift_min=neg_min, \
                                            neg_shift_max=neg_max)

        num_thread = 1 if viz else 4
        queue_capacity = 100 if viz else 3000

        image, mask, score, background = tf.train.shuffle_batch([image, mask, score, background],
                min_after_dequeue=10 if viz else 1000 , \
                batch_size = batch_size, \
                num_threads = num_thread,\
                capacity = queue_capacity, enqueue_many =False,
        )

        return image, mask, score, background

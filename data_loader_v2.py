randNumFredUsedPoking =
shift = 16.0
inpSize = 192.0
maskSize = 112.0

def crop_data(data, xc, yc, side):
    data = tf.image.crop_to_bounding_box(data, yc-side/2.0, xc-side/2.0, side , side)
    data = tf.image.random_flip_up_down(data)
    data = tf.image.random_flip_left_right(data)
    image = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [3]), [1,2,0])

    image = tf.image.resize_images(image, [inpSize, inpSize])

    mask = tf.image.resize_images(mask, [maskSize, maskSize])
    mask = (mask + 1) / 2
    mask = tf.cast(tf.transpose(mask, [3,0,1,2])[0], tf.int32)

    return image, mask


def read_decode_positive_example_poking(filename_queue, obs_shape):
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

    xc = center_max_axis[0]
    yc = center_max_axis[1]
    maxDim = center_max_axis[2]/randNumFredUsedPoking
    scale = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    scale = scale * maxDim * 224.0/(160.0*128.0)

    xc += tf.random_uniform([1], minval=-shift, maxval=shift, dtype=tf.int32)[0] * scale
    yc += tf.random_uniform([1], minval=-shift, maxval=shift, dtype=tf.int32)[0] * scale
    side = inpSize * scale

    image, mask = crop_data(data, xc, yc, side)

    return image, mask,1

import tensorflow as tf
import numpy as np


scales = tf.convert_to_tensor([2**(-1), 2**(-0.75), 2**(-0.5), 2**0.5, 2**(0.75), 2])
randNumberPokingdata = 1.75
inpSize = 192
maskSize = 112
padSize = 500
bgSize = 160

#input number all float32
def crop_data(data, xc, yc, side, sawyer_data = False):
    data = tf.image.crop_to_bounding_box(data, tf.cast(yc-side/2.0, tf.int32), tf.cast(xc-side/2.0, tf.int32), \
                                                            tf.cast(side, tf.int32) , tf.cast(side, tf.int32))
    data = tf.image.random_flip_up_down(data)
    data = tf.image.random_flip_left_right(data)
    image = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [3]), [1,2,0])

    image = tf.image.resize_images(image, [inpSize, inpSize])

    # import pdb; pdb.set_trace()
    mask = tf.image.resize_images(mask, [maskSize, maskSize])
    if not sawyer_data:
        mask = (mask + 1) / 2
    mask = tf.cast(tf.transpose(mask, [2,0,1])[0], tf.int32) #msk rank2

    return image, mask

def img_float2uint(img_tensor):
    img_tensor = (img_tensor + 0.5)*255.0
    return tf.cast(img_tensor, tf.uint8)
def read_decode_positive_example_poking(filename_queue, pos_shift, addBg, do_scale=True):
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
    maxDim = tf.clip_by_value(center_max_axis[2]/randNumberPokingdata, 0.0, 60.0)

    data = tf.cast(features['data'], tf.float32)
    data = tf.reshape(data, [240,240,4])
    background = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0]) #240x240
    data = tf.pad(data,[[padSize,padSize],[padSize,padSize],[0,0]])

    if do_scale:
        scale = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    else:
        scale = 1.0

    scale = scale * maxDim * 224.0/(160.0*128.0)
    side = scale * inpSize

    xc = center_max_axis[0] + padSize
    yc = center_max_axis[1] + padSize

    xc += tf.random_uniform([1], minval=-pos_shift, maxval=pos_shift, dtype=tf.float32)[0] * scale
    yc += tf.random_uniform([1], minval=-pos_shift, maxval=pos_shift, dtype=tf.float32)[0] * scale

    image, mask = crop_data(data, xc, yc, side)
    image = img_float2uint(image)
    mask = (mask + 1)/2
    mask = tf.cast(mask, tf.uint8)
    if addBg:
        background = tf.image.resize_images(background, [bgSize, bgSize])
        background = img_float2uint(background)
        return image, mask, tf.cast(1, tf.uint8), background
    else:
        return image, mask, tf.cast(1, tf.uint8), tf.cast(2, tf.uint8)


def read_decode_negative_example_poking(filename_queue, neg_shift_min, neg_shift_max, addBg):

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
    maxDim = tf.clip_by_value(center_max_axis[2]/randNumberPokingdata, 0.0, 60.0)

    data = tf.cast(features['data'], tf.float32)
    data = tf.reshape(data, [240,240,4])

    background = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    data = tf.pad(data,[[padSize,padSize],[padSize,padSize],[0,0]])
    #if good range, using bad shift, if bad range, using arbitrary shift
    random_index = tf.random_uniform([1], minval=0, maxval=6, dtype=tf.int32)[0]


    scale1 = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    scale1 = scale1
    scale2 = scales[random_index]

    coin_flip = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)[0]

    scale = tf.where(coin_flip < 3/(3+6), x = scale1, y = scale2) # 1/3 probability of using correct scale
    scale = scale * maxDim * 224.0/(160.0*128.0)
    side = scale * inpSize

    shift_min = tf.where(coin_flip < 3/(3+6), x = neg_shift_min, y = 0)
    shift_max = neg_shift_max

    xc = center_max_axis[0] + padSize
    yc = center_max_axis[1] + padSize

    #random from -1, 1 decide the jettering direction
    # import pdb; pdb.set_trace()
    xc += tf.cast(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)[0]*2-1, tf.float32) * \
                tf.random_uniform([1], minval=shift_min, maxval=shift_max, \
                dtype=tf.float32)[0] * scale
    yc += tf.cast(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)[0]*2-1, tf.float32) * \
                tf.random_uniform([1], minval=shift_min, maxval=shift_max, \
                dtype=tf.float32)[0] * scale

    image, mask = crop_data(data, xc, yc, side)
    image = img_float2uint(image)
    mask = (mask + 1)/2
    mask = tf.cast(mask, tf.uint8)
    if addBg:
        background = tf.image.resize_images(background, [bgSize, bgSize])
        background = img_float2uint(background)
        return image, tf.cast(2, tf.uint8), tf.cast(0, tf.uint8), background
    else:
        return image, tf.cast(2, tf.uint8), tf.cast(0, tf.uint8), tf.cast(2, tf.uint8)

def inputs_poking(filenames,
            pos_max=24,
            neg_min=50,
            neg_max=90,
            batch_size=16,
            positive=True,
            viz=False,
            addBg = False,
            do_scale_pos_scoring=True,
            bootstrap = False,
            num_heads = 5,
    ):

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=None)
        if positive:
            image, mask, score, background = read_decode_positive_example_poking(filename_queue, \
                                            pos_max, addBg, do_scale=do_scale_pos_scoring)

        else:
            image, mask, score, background = read_decode_negative_example_poking(filename_queue, \
                                            neg_min, neg_max, addBg)

        num_thread = 1 if viz else 4
        queue_capacity = 300 if viz else 5000

        image, mask, score, background = tf.train.shuffle_batch([image, mask, score, background],
                min_after_dequeue=10 if viz else 1000 , \
                batch_size = batch_size, \
                num_threads = num_thread,\
                capacity = queue_capacity, enqueue_many =False,
        )
        image = tf.cast(image, tf.float32)
        image = (image / 255.0) - 0.5
        background = tf.cast(background, tf.float32)
        background = (background / 255.0) - 0.5
        mask = tf.cast(mask, tf.int32)
        score = tf.cast(score, tf.int32)
        if not positive: #mask dequeued is [-1.0, -1.0, -1.0....]
            mask = tf.convert_to_tensor(np.zeros([batch_size, maskSize, maskSize]), dtype=tf.int32)
        
        #if not addBg, background = [-1.0, -1.0, -1.0 ......]
        
        # #hard coded by fred
        # if addBg:
        #     background = tf.image.crop_to_bounding_box(background, 40, 40, \
        #                                                     80 ,80)
        #     background = tf.image.resize_images(background, [160, 160])
        if not bootstrap:
            return image, mask, score, background
        else:
            return image, mask, score, background, np.ones([batch_size, num_heads])


#robot negative data sacling only.... no jettering needed
def read_decode_negative_sawyer_data(filename_queue,\
                                    addBg):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'img': tf.FixedLenFeature([], tf.string),
                  'background' : tf.FixedLenFeature([], tf.string),
                  'bootstrap_id':  tf.FixedLenFeature([], tf.string),
          })
    bootstrap_id = tf.decode_raw(features['bootstrap_id'], tf.uint8)
    bootstrap_id = tf.reshape(bootstrap_id, [10])
    
    
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [448,448,3])
    img = tf.cast(img, tf.float32)
    img = tf.pad(img, [[padSize,padSize],[padSize,padSize],[0,0]])

    scale = tf.pow(2.0, tf.random_uniform([1], -1, 1))[0]
    xc = yc = 224 + padSize
    side = 224.0*scale #tf.float32, TODO: make data format compatable
    img = tf.image.crop_to_bounding_box(img, tf.cast(yc-side/2.0, tf.int32), tf.cast(xc-side/2.0, tf.int32), \
                                                            tf.cast(side, tf.int32) , tf.cast(side, tf.int32))
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize_images(img, [inpSize, inpSize])
    img = tf.cast(img, tf.uint8)
    if addBg:
        background = tf.decode_raw(features['background'], tf.uint8)
        background = tf.reshape(background, [240,240,3])
        background = tf.cast(background, tf.float32)
        background = tf.image.resize_images(background, [bgSize, bgSize])
        background = tf.cast(background, tf.uint8)
        return img, tf.cast(2, tf.uint8), tf.cast(0, tf.uint8), background, bootstrap_id
    else:
        return img, tf.cast(2, tf.uint8), tf.cast(0, tf.uint8), tf.cast(2, tf.uint8), bootstrap_id

def read_decode_positive_sawyer_data(filename_queue, pos_shift,\
                                    addBg, do_scale=True):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'img': tf.FixedLenFeature([], tf.string),
                  'mask':tf.FixedLenFeature([], tf.string),
                  'background': tf.FixedLenFeature([], tf.string),
                  'bootstrap_id':  tf.FixedLenFeature([], tf.string),
          })
    bootstrap_id = tf.decode_raw(features['bootstrap_id'], tf.uint8)
    bootstrap_id = tf.reshape(bootstrap_id, [10])
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [448,448,3])
    img = tf.cast(img, tf.float32)
    mask = tf.decode_raw(features['mask'],tf.uint8)
    mask = tf.cast(mask, tf.float32)
    mask = tf.reshape(mask, [448,448,1])

    img_mask = tf.concat([img, mask],2)
    img_mask = tf.pad(img_mask, [[padSize,padSize],[padSize,padSize],[0,0]])

    if do_scale:
        scale = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    else:
        scale = 1.0

    xc = yc = 224 + padSize
    xc += tf.random_uniform([1], minval=-pos_shift, maxval=pos_shift, dtype=tf.float32)[0] * scale
    yc += tf.random_uniform([1], minval=-pos_shift, maxval=pos_shift, dtype=tf.float32)[0] * scale

    maxDim = 128
    scale = scale*maxDim*224.0/(160.0*128)
    side = inpSize * scale


    img, mask = crop_data(img_mask, xc, yc, side, sawyer_data = True)
    img = tf.cast(img, tf.uint8)
    mask = tf.cast(mask, tf.uint8)
    if addBg:
        background = tf.decode_raw(features['background'], tf.uint8)
        background = tf.reshape(background, [240,240,3])
        background = tf.cast(background, tf.float32)
        background = tf.image.resize_images(background, [bgSize, bgSize])
        background = tf.cast(background, tf.uint8)
        return img, mask, tf.cast(1, tf.uint8), background, bootstrap_id
    else:
        return img, mask, tf.cast(1, tf.uint8), tf.cast(2, tf.uint8), bootstrap_id

def read_decode_negative_from_positive_sawyer_data(filename_queue, \
                            neg_shift_min, neg_shift_max, addBg):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'img': tf.FixedLenFeature([], tf.string),
                  'mask':tf.FixedLenFeature([], tf.string),
                  'background': tf.FixedLenFeature([], tf.string),
                  'bootstrap_id':  tf.FixedLenFeature([], tf.string),
          })
    bootstrap_id = tf.decode_raw(features['bootstrap_id'], tf.uint8)
    bootstrap_id = tf.reshape(bootstrap_id, [10])
    
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [448,448,3])
    img = tf.cast(img, tf.float32)
    mask = tf.decode_raw(features['mask'],tf.uint8)
    mask = tf.cast(mask, tf.float32)
    mask = tf.reshape(mask, [448,448,1])

    img_mask = tf.concat([img, mask], 2)
    img_mask = tf.pad(img_mask, [[padSize,padSize],[padSize,padSize],[0,0]])

    scale1 = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    scale1 = scale1
    random_index = tf.random_uniform([1], minval=0, maxval=6, dtype=tf.int32)[0]
    scale2 = scales[random_index]
    coin_flip = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)[0]

    scale = tf.where(coin_flip < 3/(3+6), x = scale1, y = scale2) # 1/3 probability of using correct scale
    maxDim = 128
    scale = scale * maxDim * 224.0/(160.0*128.0)
    side = scale * inpSize

    shift_min = tf.where(coin_flip < 3/(3+6), x = neg_shift_min, y = 0)
    shift_max = neg_shift_max

    xc = 224 + padSize
    yc = 224 + padSize

    #random from -1, 1 decide the jettering direction
    xc += tf.cast(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)[0]*2-1, tf.float32) * \
                tf.random_uniform([1], minval=shift_min, maxval=shift_max, \
                dtype=tf.float32)[0] * scale
    yc += tf.cast(tf.random_uniform([1], minval=0, maxval=2, dtype=tf.int32)[0]*2-1, tf.float32) * \
                tf.random_uniform([1], minval=shift_min, maxval=shift_max, \
                dtype=tf.float32)[0] * scale
    img, mask = crop_data(img_mask, xc, yc, side)
    img = tf.cast(img, tf.uint8)
    mask = tf.cast(mask, tf.uint8)
    if addBg:
        background = tf.decode_raw(features['background'], tf.uint8)
        background = tf.reshape(background, [240,240,3])
        background = tf.cast(background, tf.float32)
        background = tf.image.resize_images(background, [bgSize, bgSize])
        background = tf.cast(background, tf.uint8)
        return img, tf.cast(2, tf.uint8), tf.cast(0, tf.uint8), background, bootstrap_id
    else:
        return img, tf.cast(2, tf.uint8), tf.cast(0, tf.uint8), tf.cast(2, tf.uint8), bootstrap_id

def inputs_sawyer_data(filenames, mode, pos_max, neg_min, jetter_max=90, train=True, \
                        batch_size=12, num_epochs=None, viz=False, addBg = False, do_scale_pos_scoring=True,\
                        bootstrap = False, num_heads = 5):
    assert (mode in set(["positive", "negative", "negative_from_positive"]))
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=None,shuffle=True)
        if mode == "positive":
            image, mask, score, background, bootstrap_id = read_decode_positive_sawyer_data(filename_queue,pos_max, addBg, do_scale=do_scale_pos_scoring)

        elif mode == "negative":
            image, mask, score, background, bootstrap_id = read_decode_negative_sawyer_data(filename_queue,addBg)
        elif mode == "negative_from_positive":
            image, mask, score, background, bootstrap_id = read_decode_negative_from_positive_sawyer_data(filename_queue,\
                                            neg_min, jetter_max, addBg)

        if train:
            num_thread = 20
            queue_capacity = 100 if viz else 5000
        else:
            num_thread = 4
            queue_capacity = 100 if viz else 5000
        image, mask, score, background, bootstrap_id = tf.train.shuffle_batch([image, mask, score, background, bootstrap_id[:num_heads]],
                                min_after_dequeue=1000 , \
                                batch_size = batch_size, \
                                num_threads = num_thread,\
                                capacity = queue_capacity, enqueue_many =False)
        #normalize everything
        image = tf.cast(image, tf.float32)
        mask = tf.cast(mask, tf.int32)
        background = tf.cast(background, tf.float32)
        image = (image / 255.0) - 0.5
        background= (background / 255.0) - 0.5
        score = tf.cast(score, tf.int32)
        # #hard coded by fred
        # if addBg:
        #     background = tf.image.crop_to_bounding_box(background, 40, 40, \
        #                                                     80 ,80)
        #     background = tf.image.resize_images(background, [160, 160])
        if mode in set(["negative", "negative_from_positive"]):
            mask = tf.convert_to_tensor(np.zeros([batch_size, maskSize, maskSize]), dtype = tf.int32)
        if not bootstrap:
            return image, mask, score, background
        else:
            return image, mask, score, background, tf.cast(bootstrap_id, tf.float32)



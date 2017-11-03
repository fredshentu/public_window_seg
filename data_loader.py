import tensorflow as tf
import numpy as np


scales = tf.convert_to_tensor([2**(-0.25),1.0, 2**(0.25), 2**(0.5), 2**0.75, 2**(-0.5), 2**(-0.75), \
                                            2**(-1.0),2**(1.0)])

def crop_random_jettering(image_tensor, middle_offset_heigh, middle_offset_width,\
                jettering_min, jettering_max, crop_size):
    jettering_min = tf.cast(jettering_min, tf.int32)
    jettering_max = tf.cast(jettering_max, tf.int32)
    elems = tf.cast(tf.convert_to_tensor([1, -1]), tf.int32)
    samples = tf.multinomial(tf.log([[10.,10.]]), 2) # note log-prob
    
    offset_width = middle_offset_width + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=jettering_min,
                                                                        maxval=jettering_max,
                                                                        dtype=tf.int32)[0]
    offset_height = middle_offset_heigh + \
                        elems[tf.cast(samples[0][1], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=jettering_min,
                                                                        maxval=jettering_max,
                                                                        dtype=tf.int32)[0]
    data = tf.image.crop_to_bounding_box(image_tensor, offset_height, offset_width,\
                                                       crop_size , crop_size)
    data = tf.image.random_flip_up_down(data)
    data = tf.image.random_flip_left_right(data)
    image = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [3]), [1,2,0])
    return image, mask

def read_decode_positive_example(filename_queue, obs_shape, jetter_length):
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
    
    scale = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    scale = scale * (224.0/long_axis)
    
    center_x = center_max_axis[0] * scale + 200
    center_y = center_max_axis[1] * scale + 200
    int_final_size = tf.cast(240*scale, tf.int32)
    data = tf.image.resize_images(data, (int_final_size, int_final_size))
    
    
    
    data = tf.pad(data, [[200,200],[200,200],[0,0]])



    middle_offset_heigh = tf.cast(center_y - 112, tf.int32)
    middle_offset_width = tf.cast(center_x - 112, tf.int32)
    
    #224 crop size, jettering should be 22 instead of 16
    jettering_min = jetter_length[0][0]
    jettering_max = jetter_length[0][1]
    image, mask = crop_random_jettering(data, middle_offset_heigh, middle_offset_width,\
                            jettering_min, jettering_max, 224)
    
    return image, mask , 1, background


def read_decode_negative_example(filename_queue, obs_shape,jetter_length):
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
    # background = tf.image.resize_images()

    
    center_x = center_max_axis[0] 
    center_y = center_max_axis[1]
    long_axis = center_max_axis[2]
    #scaling to this data
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=9,
                            dtype=tf.int32)[0]
    #if good range, using bad offset, if bad range, using arbitrary offset
   
    
    
    scale1 = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    scale1 = scale1 * (224.0/long_axis)
    #sometimes use large offset to learn about padding and arena
    scale2 = scales[random_index] * (224.0/long_axis)
    
    scale = tf.where(random_index < 3, x = scale1, y = scale2)
    
    jetter_range_index = tf.where(random_index < 3, x = 1, y = 2)
    jettering_min = jetter_length[jetter_range_index][0]
    jettering_max = jetter_length[jetter_range_index][1]
    
    
    
    center_x = center_max_axis[0] * scale + 200
    center_y = center_max_axis[1] * scale + 200
    int_final_size = tf.cast(240*scale, tf.int32)
    data = tf.image.resize_images(data, (int_final_size, int_final_size))
    
    data = tf.pad(data, [[200,200],[200,200],[0,0]])



    middle_offset_heigh = tf.cast(center_y - 112, tf.int32)
    middle_offset_width = tf.cast(center_x - 112, tf.int32)
    
    
    image, mask = crop_random_jettering(data, middle_offset_heigh, middle_offset_width,\
                            jettering_min, jettering_max, 224)

    return image, mask, 0, background

def inputs_poking(filenames, pos_max, neg_min, obs_shape = [240,240,4], train=True, \
                batch_size=16, num_epochs = None, positive = True, viz=False):
    jetter_length = tf.convert_to_tensor([(0, pos_max), (neg_min, 64), (0, 64), (64, 90)])
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs)
        if positive:
            image, mask, score, background = read_decode_positive_example(filename_queue, obs_shape, jetter_length)
        
        else:
            image, mask, score, background = read_decode_negative_example(filename_queue, obs_shape, jetter_length)
            
            
        if train:
            num_thread = 12
            queue_capacity = 100 if viz else 3000
        else:
            num_thread = 4
            queue_capacity = 100 if viz else 3000
        image, mask,score, background = tf.train.shuffle_batch([image,mask,score], 
                                min_after_dequeue=1000 , \
                                batch_size = batch_size, \
                                num_threads = num_thread,\
                                capacity = queue_capacity, enqueue_many = False)
        
        image = tf.image.resize_images(image, [160,160])
        
        downsampled_mask = tf.image.resize_images(mask, [112,112])
        downsampled_mask = (downsampled_mask + 1) / 2
        downsampled_mask = tf.cast(tf.transpose(downsampled_mask, [3,0,1,2])[0], tf.int32)
        background = tf.image.resize_images(background, [160, 160])

        return image, downsampled_mask, score, background
        
        
def read_decode_negative_sawyer_data(filename_queue, jetter_length, obs_shape = [256,256]):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'img': tf.FixedLenFeature([], tf.string),
          })
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [448,448,3])
    img = tf.cast(img, tf.float32)
    img = (img/255.0) - 0.5
    background = tf.transpose(tf.gather(tf.transpose(img, [2,0,1]), [0,1,2]), [1,2,0])
    img = tf.pad(img, [[200,200],[200,200],[0,0]])
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=9,
                            dtype=tf.int32)[0]
                            
    crop_size = scales[random_index] * 112

    middle_offset_heigh = 224 + 200 - tf.cast(crop_size, tf.int32)
    middle_offset_width = 224 + 200 - tf.cast(crop_size, tf.int32)
    
    
    
    int_crop_size = tf.cast(scales[random_index] * 224, tf.int32)
    img = tf.image.crop_to_bounding_box(img, middle_offset_heigh, middle_offset_width,\
                                                       int_crop_size , int_crop_size)

    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize_images(img, [224,224])
    
    return img,tf.convert_to_tensor(np.zeros([224,224,1])), 0, background


def read_decode_negative_from_positive_sawyer_data(filename_queue, jetter_length, obs_shape = [256,256]):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'img': tf.FixedLenFeature([], tf.string),
                  'mask':tf.FixedLenFeature([], tf.string),
          })
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [448,448,3])
    img = tf.cast(img, tf.float32)
    img = (img/255.0) - 0.5
    background = tf.transpose(tf.gather(tf.transpose(img, [2,0,1]), [0,1,2]), [1,2,0])

    mask = tf.decode_raw(features['mask'],tf.uint8)
    mask = tf.cast(mask, tf.float32)
    mask = tf.reshape(mask, [448,448,1])
    
    img_mask = tf.concat(2,[img, mask])
    img_mask = tf.pad(img_mask, [[200,200],[200,200],[0,0]])
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=9,
                            dtype=tf.int32)[0]
    scale1 = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    scale2 = scales[random_index]
    scale = tf.where(random_index < 3, x = scale1, y = scale2)
    
    jetter_range_index = tf.where(random_index < 3, x = 1, y = 2)
    jettering_min = jetter_length[jetter_range_index][0]*scale
    jettering_max = jetter_length[jetter_range_index][1]*scale
    middle_offset_heigh = 200 + 224 - tf.cast(scale * 112, tf.int32)
    middle_offset_width = 200 + 224 - tf.cast(scale * 112, tf.int32)
    
    
    image, mask = crop_random_jettering(img_mask, middle_offset_heigh, middle_offset_width,\
                            jettering_min, jettering_max, tf.cast(scale * 224, tf.int32))
    
    image = tf.image.resize_images(image, [224,224])
    mask = tf.image.resize_images(mask, [224,224])
    
    return image, mask, 0, background
    
    
def read_decode_positive_example_sawyer_data(filename_queue, jetter_length, obs_shape = [256,256]):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
                  'img': tf.FixedLenFeature([], tf.string),
                  'mask':tf.FixedLenFeature([], tf.string),
          })
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, [448,448,3])
    img = tf.cast(img, tf.float32)
    img = (img/255.0) - 0.5
    background = tf.transpose(tf.gather(tf.transpose(img, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.decode_raw(features['mask'],tf.uint8)
    mask = tf.cast(mask, tf.float32)
    mask = tf.reshape(mask, [448,448,1])
    
    img_mask = tf.concat(2,[img, mask])
    img_mask = tf.pad(img_mask, [[200,200],[200,200],[0,0]])
    
    scale = tf.pow(2.0, tf.random_uniform([1], -0.25, 0.25))[0]
    
    middle_offset_heigh = 200+224 - tf.cast(scale * 112, tf.int32)
    middle_offset_width = 200+224 - tf.cast(scale * 112, tf.int32)
    jettering_min = jetter_length[0][0]*scale
    jettering_max = jetter_length[0][1]*scale
    
    image, mask = crop_random_jettering(img_mask, middle_offset_heigh, middle_offset_width,\
                            jettering_min, jettering_max, tf.cast(scale * 224, tf.int32))
    
    image = tf.image.resize_images(image, [224,224])
    mask = tf.image.resize_images(mask, [224,224])
    
    return image, mask, 1, background


def inputs_sawyer_data(filenames, mode, pos_max, neg_min,train=True, batch_size=12, num_epochs = None, viz=False):
    jetter_length = tf.cast(tf.convert_to_tensor([(0, pos_max), (neg_min, 64), (0, 64), (64, 90)]), tf.float32)
    
    assert (mode in set(["positive", "negative", "negative_from_positive"]))
    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs, shuffle=True)
        if mode == "positive":
            image, mask, score, background = read_decode_positive_example_sawyer_data(filename_queue,jetter_length)
        
        elif mode == "negative":
            image, mask, score, background = read_decode_negative_sawyer_data(filename_queue,jetter_length)
        elif mode == "negative_from_positive":
            image, mask, score, background = read_decode_negative_from_positive_sawyer_data(filename_queue,jetter_length)
        
        if train:
            num_thread = 20
            queue_capacity = 100 if viz else 3000
        else:
            num_thread = 4
            queue_capacity = 100 if viz else 3000
        image, mask, score, background = tf.train.shuffle_batch([image, mask, score, background], 
                                min_after_dequeue=1000 , \
                                batch_size = batch_size, \
                                num_threads = num_thread,\
                                capacity = queue_capacity, enqueue_many =False)
        
        image = tf.image.resize_images(image, [160,160])
        
        mask = tf.image.resize_images(mask, [112,112])
        mask = tf.cast(tf.transpose(mask, [3,0,1,2])[0], tf.int32)
        background = tf.image.resize_images(background, [160, 160])
        
        return image, mask, score, background
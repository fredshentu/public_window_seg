import tensorflow as tf

pos_scale_up = 2^(0.25)
pos_scale_down = 2^(-0.25)
neg_scales = [2^(i*0.25 - 3) for i in range(17)] #index 10, index 11, index 12 good scale 


def read_decode_positive_example(filename_queue, obs_shape):
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
    
    center_x = center_max_axis[0] 
    center_y = center_max_axis[1]
    long_axis = center_max_axis[2]
    
    
    #scaling to this data
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=8,
                            dtype=tf.int32)[0]
    scale = scales[random_index] * (224.0/long_axis)
    center_x = center_max_axis[0] * scale + 200
    center_y = center_max_axis[1] * scale + 200
    int_final_size = tf.cast(240*scale, tf.int32)
    data = tf.image.resize_images(data, (int_final_size, int_final_size))
    
    
    
    data = tf.pad(data, [[200,200],[200,200],[0,0]])



    middle_offset_heigh = tf.cast(center_y - 112, tf.int32)
    middle_offset_width = tf.cast(center_x - 112, tf.int32)
    
    
    elems = tf.cast(tf.convert_to_tensor([1, -1]), tf.int32)
    samples = tf.multinomial(tf.log([[10.,10.]]), 1) # note log-prob
    offset_width = middle_offset_width + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=jetter_length[0][0],
                                                                        maxval=jetter_length[0][1],
                                                                        dtype=tf.int32)[0]
    offset_height = middle_offset_heigh + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=jetter_length[0][0],
                                                                        maxval=jetter_length[0][1],
                                                                        dtype=tf.int32)[0]
    data = tf.image.crop_to_bounding_box(data, offset_height, offset_width,\
                                                       224 , 224)
    data = tf.image.random_flip_up_down(data)
    data = tf.image.random_flip_left_right(data)
    image = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [3]), [1,2,0])
    
    return image, mask,1


def read_decode_negative_example(filename_queue, obs_shape):
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
    
    center_x = center_max_axis[0] 
    center_y = center_max_axis[1]
    long_axis = center_max_axis[2]
    #scaling to this data
    random_index = tf.random_uniform(
                            [1],
                            minval=0,
                            maxval=5,
                            dtype=tf.int32)[0]
    #if good range, using bad offset, if bad range, using arbitrary offset
    jetter_range_index = tf.where(random_index < 3, x = 1, y = 2)
    #sometimes use large offset to learn about padding and arena
    
    jetter_range_index_final = tf.where(random_index < 2, x = 3, y = jetter_range_index)
    min_jettering = jetter_length[jetter_range_index_final][0]
    max_jettering = jetter_length[jetter_range_index_final][1]
    
    
    
    
    scale = scales[random_index] * (224.0/long_axis)
    center_x = center_max_axis[0] * scale + 200
    center_y = center_max_axis[1] * scale + 200
    int_final_size = tf.cast(240*scale, tf.int32)
    data = tf.image.resize_images(data, (int_final_size, int_final_size))
    
    
    
    data = tf.pad(data, [[200,200],[200,200],[0,0]])



    middle_offset_heigh = tf.cast(center_y - 112, tf.int32)
    middle_offset_width = tf.cast(center_x - 112, tf.int32)
    
    
    elems = tf.cast(tf.convert_to_tensor([1, -1]), tf.int32)
    samples = tf.multinomial(tf.log([[10.,10.]]), 1) # note log-prob
    offset_width = middle_offset_width + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=min_jettering,
                                                                        maxval=max_jettering,
                                                                        dtype=tf.int32)[0]
    offset_height = middle_offset_heigh + \
                        elems[tf.cast(samples[0][0], tf.int32)]*tf.random_uniform(
                                                                        [1],
                                                                        minval=min_jettering,
                                                                        maxval=max_jettering,
                                                                        dtype=tf.int32)[0]
    data = tf.image.crop_to_bounding_box(data, offset_height, offset_width,\
                                                       224 , 224)
    data = tf.image.random_flip_up_down(data)
    data = tf.image.random_flip_left_right(data)
    image = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [0,1,2]), [1,2,0])
    mask = tf.transpose(tf.gather(tf.transpose(data, [2,0,1]), [3]), [1,2,0])

    return image, mask, 0

def inputs(filenames, obs_shape, train=True, batch_size=16, num_epochs = None, positive = True):

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs)
        if positive:
            image, mask, score= read_decode_positive_example(filename_queue, obs_shape)
        
        else:
            image, mask, score = read_decode_negative_example(filename_queue, obs_shape)
            
            
        if train:
            num_thread = 12
            queue_capacity = 3000
        else:
            num_thread = 4
            queue_capacity = 3000
        image, mask,score = tf.train.shuffle_batch([image,mask,score], 
                                min_after_dequeue=1000 , \
                                batch_size = batch_size, \
                                num_threads = num_thread,\
                                capacity = queue_capacity, enqueue_many = False)
        
        image = tf.image.resize_images(image, [160,160])
        
        downsampled_mask = tf.image.resize_images(mask, [112,112])
        downsampled_mask = (downsampled_mask + 1) / 2
        downsampled_mask = tf.cast(tf.transpose(downsampled_mask, [3,0,1,2])[0], tf.int32)

        return image, downsampled_mask, score

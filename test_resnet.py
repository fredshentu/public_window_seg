import tensorflow as tf
slim = tf.contrib.slim
from resnet import resnet_v1
import numpy as np
from scipy.misc import imread, imresize

checkpoint_file = './data/resnet-50/resnet_v1_50.ckpt'
sample_images = ['dog.jpg', 'panda.jpg']
#Load the model
sess = tf.Session()
inputs = tf.placeholder(tf.float32, [None, 160,160,3])
background = tf.placeholder(tf.float32, [None, 160,160,3])
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_50(inputs, background, 1000, is_training=False)
saver = tf.train.Saver()
saver.restore(sess, checkpoint_file)

# import pdb; pdb.set_trace()
# def get_candidate(result, topn=5):
# 	with open('./imagenet.txt') as f:
# 		dictionary = eval(f.read())
# 	idxes = np.argsort(result[0])[-topn:][::-1]
# 	results = []
# 	for idx in idxes:
# 		results.append(dictionary[idx])
# 	return results

# imgs = ['./cat.jpg', './car.jpeg', './house.jpeg']
# for img_path in imgs:
# 	img = imresize(imread(img_path), [224,224])
# 	results = sess.run(net, { inputs : [img] })
# 	print (get_candidate(results))
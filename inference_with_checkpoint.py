import tensorflow as tf
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from nets import nets_factory
import numpy as np
import os

slim = tf.contrib.slim
from nets.inception_resnet_v2 import *

tf.app.flags.DEFINE_string('dataset_name', 'cifar10', '')
tf.app.flags.DEFINE_string('dataset_dir', 'F:\\002---study\\00AA_AI\\CSDN\\tmp\\cifar10', '')
tf.app.flags.DEFINE_string('model_name', 'cifarnet', '')
tf.app.flags.DEFINE_string('output_file', 'F:\\002---study\\00AA_AI\\CSDN\\tmp\\cifar10\\output.pb', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'F:\\002---study\\00AA_AI\\CSDN\\tmp\\cifarnet-model', '')
tf.app.flags.DEFINE_string('pic_path', 'F:\\002---study\\00AA_AI\\CSDN\\tmp\\pic_path\\testcat.jpg', '')

FLAGS = tf.app.flags.FLAGS
is_training = False
preprocessing_name = FLAGS.model_name

graph = tf.Graph().as_default()

sess = tf.Session()

dataset = dataset_factory.get_dataset(
    FLAGS.dataset_name, 'train', FLAGS.dataset_dir)

image_preprocessing_fn = preprocessing_factory.get_preprocessing(
    preprocessing_name,
    is_training=False)

network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes=dataset.num_classes,
    is_training=False)

if hasattr(network_fn, 'default_image_size'):
    image_size = network_fn.default_image_size
else:
    image_size = FLAGS.default_image_size

placeholder = tf.placeholder(tf.string, name='input')
image = tf.image.decode_jpeg(placeholder, channels=3)
image = image_preprocessing_fn(image, image_size, image_size)
image = tf.expand_dims(image, 0)
logit, end_points = network_fn(image)

saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
saver.restore(sess, checkpoint_path)
image_value = open(FLAGS.pic_path, 'rb').read()
logit_value = sess.run([logit], feed_dict={placeholder: image_value})
print(logit_value)
print(np.argmax(logit_value))

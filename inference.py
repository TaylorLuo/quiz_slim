import tensorflow as tf
import os
import sys
slim = tf.contrib.slim
from nets.inception_resnet_v2 import *
from preprocessing import preprocessing_factory
from nets import nets_factory
import numpy as np
from datasets import dataset_factory


FLAGS = tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string('output_folder', '', '')
tf.app.flags.DEFINE_string('input_checkpoint', 'F:\\002---study\\00AA_AI\\CSDN\\tmp\\cifarnet-model','')
tf.app.flags.DEFINE_string('dataset_name', 'cifar10', 'The name of the dataset to load.')
tf.app.flags.DEFINE_string('dataset_split_name', 'validation', 'The name of the train/test split.')
tf.app.flags.DEFINE_string('dataset_dir', 'F:\\002---study\\00AA_AI\\CSDN\\tmp\\pic_path', 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('file_name', 'test.jpg', 'The name of the file.')
tf.app.flags.DEFINE_string('model_name', 'cifar10', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')


sample_images = FLAGS.file_name.split(',')
# Load the model
sess = tf.Session()

dataset = dataset_factory.get_dataset(
    FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

# Select the model #
####################
network_fn = nets_factory.get_network_fn(
    FLAGS.model_name,
    num_classes=(dataset.num_classes - FLAGS.labels_offset),
    is_training=False)


def eval():
    eval_image_size = 299
    num_classes = dataset.num_classes

    preprocessing_name = FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
    input_tensor = tf.placeholder(tf.string, name='DecodeJpeg/contents')
    image = tf.image.decode_jpeg(input_tensor, channels=3)
    image = image_preprocessing_fn(image,
                                   0, 0)
#     eval_image_size, eval_image_size)
    image = tf.expand_dims(image, 0)
    logits, end_points = network_fn(image)
    prediction = tf.nn.softmax(logits, name='prediction')
    saver = tf.train.Saver()
    if(os.path.isdir(FLAGS.input_checkpoint)):
        ckpt_filename = tf.train.latest_checkpoint(FLAGS.input_checkpoint)
    else:
        ckpt_filename = FLAGS.input_checkpoint
    print('restored from file %s' % ckpt_filename)
    return logits, prediction, end_points


logits, prediction, end_points = eval()
for image_name in sample_images:
    with open(image_name, 'rb') as f:
        image_data = f.read()
    train_folder, checkpoint = os.path.split(FLAGS.input_checkpoint)
    logit_values, predict_values, end_points_values = sess.run(
        [logits, prediction, end_points],
        feed_dict={'DecodeJpeg/contents:0': image_data})
    print(np.shape(logit_values))
    print(np.shape(end_points_values['input']))
    _, h, w, c = np.shape(predict_values)
    for x in range(h):
        for y in range(w):
            pred = predict_values[0, x, y, :]
            print(np.max(pred))
            print(np.argmax(pred))

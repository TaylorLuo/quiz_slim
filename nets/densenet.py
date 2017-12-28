"""Contains a variant of the densenet model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def trunc_normal(stddev): return tf.truncated_normal_initializer(stddev=stddev)


def bn_act_conv_drp(current, num_outputs, kernel_size, scope='block'):
    current = slim.batch_norm(current, scope=scope + '_bn')
    current = tf.nn.relu(current)
    current = slim.conv2d(current, num_outputs, kernel_size, scope=scope + '_conv')
    current = slim.dropout(current, scope=scope + '_dropout')
    return current


def block(net, layers, growth, scope='block'):
    for idx in range(layers):
        bottleneck = bn_act_conv_drp(net, 4 * growth, [1, 1],
                                     scope=scope + '_conv1x1' + str(idx))
        tmp = bn_act_conv_drp(bottleneck, growth, [3, 3],
                              scope=scope + '_conv3x3' + str(idx))
        net = tf.concat(axis=3, values=[net, tmp])
    return net


"""
do convolution and pooling
consists: BN-Conv(1X1)-Pool(2X2)
"""


def transition_layer(net, growth, scope='transition'):
    net = bn_act_conv_drp(net, growth, [1, 1], scope=scope + '_conv1x1' + str(0))
    net = slim.avg_pool2d(net, [2, 2], stride=2, padding='VALID')
    return net


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
    return kernel_size_out


def densenet(images, num_classes=1001, is_training=False,
             dropout_keep_prob=0.8,
             scope='densenet'):
    """Creates a variant of the densenet model.

      images: A batch of `Tensors` of size [batch_size, height, width, channels].
      num_classes: the number of classes in the dataset.
      is_training: specifies whether or not we're currently training the model.
        This variable will determine the behaviour of the dropout layer.
      dropout_keep_prob: the percentage of activation values that are retained.
      prediction_fn: a function to get predictions out of logits.
      scope: Optional variable_scope.

    Returns:
      logits: the pre-softmax activations, a tensor of size
        [batch_size, `num_classes`]
      end_points: a dictionary from components of the network to the corresponding
        activation.
    """
    growth = 24
    compression_rate = 0.5

    def reduce_dim(input_feature):
        return int(int(input_feature.shape[-1]) * compression_rate)

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            # init convolution 224x224x3f-->112x112x48f
            end_point = 'Conv_0_2g_3x3'
            net = slim.conv2d(images, 2 * growth, [7, 7], stride=2, scope=end_point)
            end_points[end_point] = net
            # init pooling  112x112x48f-->56x56x48f
            end_point = 'Pool_0_2d_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point, padding='SAME')
            end_points[end_point] = net
            # dense_1  56x56x48f-->56x56x96f-->56x56x24f   *6
            net = block(net, 6, growth, scope='dense_1')
            # trans_1  56x56x24f-->28x28x24f
            net = transition_layer(net, growth, scope='trans_1')
            # dense_2  28x28x24f-->28x28x96f-->28x28x24f   *12
            net = block(net, 12, growth, scope='dense_2')
            # trans_2  28x28x24f-->14x14x24f
            net = transition_layer(net, growth, scope='trans_2')
            # dense_3  14x14x24f-->14x14x96f-->14x14x24f   *48
            net = block(net, 48, growth, scope='dense_3')
            # trans_3  14x14x24f-->7x7x24f
            net = transition_layer(net, growth, scope='trans_3')
            # dense_final 7x7x24f  *32
            net = block(net, 32, growth, scope='dense_final')
            kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
            # 1x1x24f
            net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                  scope='AvgPool_1a_{}x{}'.format(*kernel_size))
            end_points['AvgPool_1a'] = net
            net = slim.dropout(net, scope=scope + '_dropout')
            end_points['PreLogits'] = net
            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                 normalizer_fn=None, scope='Conv2d_1c_1x1')
            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
            end_points['Logits'] = logits
            end_points['Predictions'] = slim.softmax(logits, scope='Predictions')
    return logits, end_points


def bn_drp_scope(is_training=True, keep_prob=0.8):
    keep_prob = keep_prob if is_training else 1
    with slim.arg_scope(
            [slim.batch_norm],
            scale=True, is_training=is_training, updates_collections=None):
        with slim.arg_scope(
                [slim.dropout],
                is_training=is_training, keep_prob=keep_prob) as bsc:
            return bsc


def densenet_arg_scope(weight_decay=0.004):
    """Defines the default densenet argument scope.

    Args:
      weight_decay: The weight decay to use for regularizing the model.

    Returns:
      An `arg_scope` to use for the inception v3 model.
    """
    with slim.arg_scope(
            [slim.conv2d],
            weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode='FAN_IN', uniform=False),
            activation_fn=None, biases_initializer=None, padding='same',
            stride=1) as sc:
        return sc


densenet.default_image_size = 224

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


compression_rate = 0.5


def reduce_dim(input_feature):
    return int(int(input_feature.shape[-1]) * compression_rate)


"""
do convolution and pooling
consists: BN-Conv(1X1)-Pool(2X2)
"""


def transition_layer(net, scope='transition'):
    net = bn_act_conv_drp(net, reduce_dim(net), [1, 1], scope=scope + '_conv1x1' + str(0))
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

    end_points = {}

    with tf.variable_scope(scope, 'DenseNet', [images, num_classes]):
        with slim.arg_scope(bn_drp_scope(is_training=is_training,
                                         keep_prob=dropout_keep_prob)) as ssc:
            # 格式化数据
            images = tf.reshape(images, [-1, 224, 224, 3])
            # init convolution 224x224x3f-->112x112x48f
            end_point = 'Conv_0_2g_7x7'
            net = slim.conv2d(images, 2 * growth, [7, 7], stride=2, scope=end_point)
            end_points[end_point] = net

            # init pooling  112x112x48f-->56x56x48f
            end_point = 'Pool_0_2d_3x3'
            net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point, padding='SAME')
            end_points[end_point] = net

            # dense_1  56x56x48f-->56x56x192f  (48+24*6)
            end_point = 'dense_1'
            net = block(net, 6, growth, scope=end_point)
            end_points[end_point] = net

            # trans_1  56x56x192f-->56x56x96f-->28x28x96f
            end_point = 'trans_1'
            net = transition_layer(net, scope=end_point)
            end_points[end_point] = net

            # dense_2  28x28x96f-->28x28x384f    (96+24*12)
            end_point = 'dense_2'
            net = block(net, 12, growth, scope=end_point)
            end_points[end_point] = net

            # trans_2  28x28x384f-->28x28x192f-->14x14x192f
            end_point = 'trans_2'
            net = transition_layer(net, scope=end_point)
            end_points[end_point] = net

            # dense_3  14x14x192f-->14x14x1344f   (192+24*48)
            end_point = 'dense_3'
            net = block(net, 48, growth, scope=end_point)
            end_points[end_point] = net

            # trans_3  14x14x1344f-->14x14x672f-->7x7x672f
            end_point = 'trans_3'
            net = transition_layer(net, scope=end_point)
            end_points[end_point] = net

            # dense_final 7x7x672f-->7x7x1440f  (672+24*32)
            end_point = 'dense_final'
            net = block(net, 32, growth, scope=end_point)
            end_points[end_point] = net

            # 1.乘以固定尺寸 来变为bx1x1x1440
            # kernel_size = _reduced_kernel_size_for_small_input(net, [7, 7])
            # # 1x1x1440f
            # net = slim.avg_pool2d(net, kernel_size, padding='VALID',
            #                       scope='AvgPool_1a_{}x{}'.format(*kernel_size))
            # end_points['AvgPool_1a'] = net

            # 2.全局平均池化 来变为bx1x1x1440f
            end_point = 'Global_avg_pooling'
            net = tf.reduce_mean(net, [1, 2], keep_dims=True, name=end_point)
            end_points[end_point] = net

            # 3.全局平均池化方法二   from tflearn.layers.conv import global_avg_pool
            # global_avg_pool(x, name='Global_avg_pooling')

            # 全链接层
            # 该全链接层具有1000神经元
            # 输入Tensor维度: [batch_size, 1x1x1440]
            # 输出Tensor维度: [batch_size, 1x1x1000]
            net = tf.layers.dense(inputs=net, units=1000, activation=tf.nn.relu)

            # 对全链接层的数据加入dropout操作，防止过拟合
            end_point = 'dropout'
            net = slim.dropout(net, scope=end_point)
            end_points[end_point] = net

            # Logits层，对dropout层的输出Tensor，执行分类操作
            logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')

            logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')


            # layers实现方式
            # 变为[batch_size, 7 * 7 * 1440]方法
            # net = flatten(net)   等同于tf.reshape(net, [-1, 7 * 7 * 1440])

            # 全链接层
            # 输入Tensor维度: [batch_size, 7 * 7 * 1440]
            # 输出Tensor维度: [batch_size, 7 * 7 * 1000]
            # net = tf.layers.dense(inputs=net, units=1000, activation=tf.nn.relu)

            # 对全链接层的数据加入dropout操作，防止过拟合
            # 略

            # Logits层，对dropout层的输出Tensor，执行分类操作
            # logits = tf.layers.dense(inputs=net, units=num_classes)

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

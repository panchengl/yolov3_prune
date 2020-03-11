# coding=utf-8
# for better understanding about yolov3 architecture, refer to this website (in Chinese):
# https://blog.csdn.net/leviopku/article/details/82660381

from __future__ import division, print_function

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

from utils.layer_utils import conv2d, darknet53_body, yolo_block, upsample_layer, yolo_block_pecentage


def conv2d(inputs, filters, kernel_size, strides=1):
    def _fixed_padding(inputs, kernel_size):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                        [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs

    if strides > 1:
        inputs = _fixed_padding(inputs, kernel_size)
    inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides,
                         padding=('SAME' if strides == 1 else 'VALID'))
    return inputs

def parse_include_res_darknet53_body_prune_factor_shortcut_list(inputs, prune_factor, shortcut_list, prune_cnt=1):
    import numpy as np
    def res_block(inputs, filters, prune_factor, prune_cnt):
        true_filters_1 = filters
        true_filters_2 = filters * 2
        for i in range(prune_cnt):
            true_filters_1 = np.floor(true_filters_1 * prune_factor)
            # true_filters_2 = np.floor(true_filters_2 * prune_factor)
        shortcut = inputs
        net = conv2d(inputs, true_filters_1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net

    # first two conv2d layers
    true_filters_conv0 = 32
    for i in range(prune_cnt):
        true_filters_conv0 = np.floor(true_filters_conv0 * prune_factor)
    net = conv2d(inputs, true_filters_conv0, 3, strides=1)
    net = conv2d(net, 64 , 3, strides=2)

    # res_block * 1
    for i in range(shortcut_list[0]):
        net = res_block(net, 32, prune_factor, prune_cnt=prune_cnt)

    net = conv2d(net, 128 , 3, strides=2)

    # res_block * 2
    for i in range(shortcut_list[1]):
        net = res_block(net, 64, prune_factor, prune_cnt=prune_cnt)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(shortcut_list[2]):
        net = res_block(net, 128, prune_factor, prune_cnt=prune_cnt)
    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(shortcut_list[3]):
        net = res_block(net, 256, prune_factor, prune_cnt=prune_cnt)
    route_2 = net
    net = conv2d(net, 1024 , 3, strides=2)

    # res_block * 4
    for i in range(shortcut_list[4]):
        net = res_block(net, 512, prune_factor, prune_cnt=prune_cnt)
    route_3 = net

    return route_1, route_2, route_3

def parse_exclude_res_darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters // 2, 1)
        net = conv2d(net, filters * 2 , 3)

        net = net + shortcut

        return net
    def darknet_last_res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters, 1)
        net = conv2d(net, filters *2, 3)
        net = net + shortcut
        return net

    # first two conv2d layers
    net = conv2d(inputs, 32 // 2, 3, strides=1)
    net = conv2d(net, 64 , 3, strides=2)

    # res_block * 1
    net = res_block(net, 32)

    net = conv2d(net, 128 , 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512 , 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    # net = darknet_last_res_block(net, 512)
    # net = conv2d(net, 1024, 1)
    route_3 = net

    return route_1, route_2, route_3

def parse_include_res_darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters // 2, 1)
        net = conv2d(net, filters * 2 // 2, 3)

        net = net + shortcut

        return net
    def darknet_last_res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters, 1)
        net = conv2d(net, filters *2, 3)
        net = net + shortcut
        return net

    # first two conv2d layers
    net = conv2d(inputs, 32 // 2, 3, strides=1)
    net = conv2d(net, 64 // 2 , 3, strides=2)

    # res_block * 1
    net = res_block(net, 32)

    net = conv2d(net, 128 // 2, 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256 // 2, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512 // 2, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024 // 2, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    net = darknet_last_res_block(net, 512)
    net = conv2d(net, 1024, 1)
    route_3 = net

    return route_1, route_2, route_3

def parse_include_res_darknet53_body_prune_factor(inputs, prune_factor, prune_cnt=1):
    import numpy as np
    def res_block(inputs, filters, prune_factor, prune_cnt):
        true_filters_1 = filters
        true_filters_2 = filters * 2
        for i in range(prune_cnt):
            true_filters_1 = np.floor(true_filters_1 * prune_factor)
            # true_filters_2 = np.floor(true_filters_2 * prune_factor)
        shortcut = inputs
        net = conv2d(inputs, true_filters_1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net

    # first two conv2d layers
    true_filters_conv0 = 32
    for i in range(prune_cnt):
        true_filters_conv0 = np.floor(true_filters_conv0 * prune_factor)
    net = conv2d(inputs, true_filters_conv0, 3, strides=1)
    net = conv2d(net, 64 , 3, strides=2)

    # res_block * 1
    net = res_block(net, 32, prune_factor, prune_cnt=prune_cnt)

    net = conv2d(net, 128 , 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64, prune_factor, prune_cnt=prune_cnt)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128, prune_factor, prune_cnt=prune_cnt)
    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256, prune_factor, prune_cnt=prune_cnt)
    route_2 = net
    net = conv2d(net, 1024 , 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512, prune_factor, prune_cnt=prune_cnt)
    route_3 = net

    return route_1, route_2, route_3

class sliming_yolov3(object):

    def __init__(self, class_num, anchors, use_label_smooth=False, use_focal_loss=False, batch_norm_decay=0.999,
                 weight_decay=5e-4):

        # self.anchors = [[10, 13], [16, 30], [33, 23],
        # [30, 61], [62, 45], [59,  119],
        # [116, 90], [156, 198], [373,326]]
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay

    def forward(self, inputs, is_training=False, reuse=False):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = parse_exclude_res_darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def forward_include_res(self, inputs, is_training=False, reuse=False):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = parse_include_res_darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def forward_include_res_with_prune_factor(self, inputs, prune_factor, is_training=False, reuse=False, prune_cnt=1):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = parse_include_res_darknet53_body_prune_factor(inputs, prune_factor, prune_cnt)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block_pecentage(route_3, 512, prune_factor, prune_cnt=prune_cnt)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block_pecentage(concat1, 256, prune_factor, prune_cnt=prune_cnt)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block_pecentage(concat2, 128, prune_factor, prune_cnt=prune_cnt)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

    def forward_include_res_with_prune_factor_get_result_no_nms(self, inputs, prune_factor, is_training=False,
                                                                reuse=False, prune_cnt=1):
        # the input img_size, form: [height, weight]
        # it will be used later
        STRIDES = [8, 16, 32]
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = parse_include_res_darknet53_body_prune_factor(inputs, prune_factor,
                                                                                              prune_cnt)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block_pecentage(route_3, 512, prune_factor, prune_cnt=prune_cnt)
                    conv_lbbox = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                             stride=1, normalizer_fn=None,
                                             activation_fn=None, biases_initializer=tf.zeros_initializer())
                    conv_lbbox = tf.identity(conv_lbbox, name='conv_lbbox')
                    # pred_lbbox = self.predict_single_featuremap(conv_lbbox, self.img_size, self.anchors[6:9])
                    # pred_lbbox = self.predict_single_featuremap(conv_lbbox, self.img_size, self.anchors[6:9])
                    pred_lbbox = self.decode_validate_doctor_yang(conv_lbbox, self.class_num, stride=STRIDES[2],
                                                                  shape=608 // 32)
                    pred_lbbox = tf.identity(pred_lbbox, name='pred_lbbox')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block_pecentage(concat1, 256, prune_factor, prune_cnt=prune_cnt)
                    conv_mbbox = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                             stride=1, normalizer_fn=None,
                                             activation_fn=None, biases_initializer=tf.zeros_initializer())
                    conv_mbbox = tf.identity(conv_mbbox, name='conv_mbbox')
                    # pred_mbbox = self.predict_single_featuremap(conv_mbbox, self.img_size, self.anchors[3:6])
                    pred_mbbox = self.decode_validate_doctor_yang(conv_mbbox, self.class_num, stride=STRIDES[1],
                                                                  shape=608 // 16)
                    pred_mbbox = tf.identity(pred_mbbox, name='pred_mbbox')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block_pecentage(concat2, 128, prune_factor, prune_cnt=prune_cnt)
                    conv_sbbox = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                             stride=1, normalizer_fn=None,
                                             activation_fn=None, biases_initializer=tf.zeros_initializer())
                    conv_sbbox = tf.identity(conv_sbbox, name='conv_sbbox')
                    pred_sbbox = self.decode_validate_doctor_yang(conv_sbbox, self.class_num, stride=STRIDES[0],
                                                                  shape=608 // 8)
                    pred_sbbox = tf.identity(pred_sbbox, name='pred_sbbox')

                    pred_sbbox = tf.reshape(pred_sbbox, (-1, self.class_num + 5))
                    pred_mbbox = tf.reshape(pred_mbbox, (-1, self.class_num + 5))
                    pred_lbbox = tf.reshape(pred_lbbox, (-1, self.class_num + 5))
                    pred_bbox = tf.concat([pred_sbbox, pred_mbbox, pred_lbbox], axis=0)
                    pred_bbox = tf.identity(pred_bbox, name='pred_bbox')
            # return conv_sbbox, conv_mbbox, conv_lbbox, pred_sbbox, pred_mbbox, pred_lbbox
            return pred_bbox

    def forward_include_res_with_prune_factor_docktor_yang(self, inputs, prune_factor, is_training=False,
                                                           reuse=False, prune_cnt=1):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = parse_include_res_darknet53_body_prune_factor(inputs, prune_factor,
                                                                                              prune_cnt)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block_pecentage(route_3, 512, prune_factor, prune_cnt=prune_cnt)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block_pecentage(concat1, 256, prune_factor, prune_cnt=prune_cnt)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block_pecentage(concat2, 128, prune_factor, prune_cnt=prune_cnt)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

                    pred_feature_maps = feature_map_1, feature_map_2, feature_map_3
                    pred_boxes, pred_confs, pred_probs = self.predict_no_stride(pred_feature_maps)
                    pred_boxes = tf.identity(pred_boxes, name='pred_boxes')
                    pred_confs = tf.identity(pred_confs, name='pred_confs')
                    pred_probs = tf.identity(pred_probs, name='pred_probs')
                    pred_boxes_last = tf.concat([pred_boxes, pred_confs, pred_probs], axis=-1)
                    pred_boxes_last = tf.reshape(pred_boxes_last, [-1, 5 + self.class_num], name='pred_boxes_last')
            return pred_boxes_last

    def decode_validate_doctor_yang(self, conv_output, num_classes, stride, shape):
        """
        :param conv_output: yolo的输出，shape为(batch_size, output_size, output_size, gt_per_grid * (5 + num_classes))
        :param num_classes: 类别的数量
        :param stride: YOLO的stride
        :return:
        pred_bbox: shape为(batch_size, output_size, output_size, gt_per_grid, 5 + num_classes)
        5 + num_classes指的是预测bbox的(xmin, ymin, xmax, ymax, confidence, probability)
        其中(xmin, ymin, xmax, ymax)是预测bbox的左上角和右下角坐标，大小是相对于input_size的，
        confidence是预测bbox属于物体的概率，probability是条件概率分布
        """
        conv_output = tf.reshape(conv_output, (1, shape, shape, 3, 5 + num_classes))
        conv_raw_dx1dy1, conv_raw_dx2dy2, conv_raw_conf, conv_raw_prob = tf.split(conv_output,
                                                                                  [2, 2, 1, num_classes],
                                                                                  axis=4)

        # y = tf.tile(tf.range(shape, dtype=tf.int32)[:, tf.newaxis], [1, shape])
        y = tf.tile(tf.expand_dims(tf.range(shape, dtype=tf.int32), 1), [1, shape])
        x = tf.tile(tf.expand_dims(tf.range(shape, dtype=tf.int32), 0), [shape, 1])
        xy_grid = tf.stack([x, y], axis=2)
        xy_grid = tf.expand_dims(xy_grid, 0)
        xy_grid = tf.expand_dims(xy_grid, 3)
        xy_grid = tf.tile(xy_grid, [1, 1, 1, 3, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)
        pred_xymin = (xy_grid + 0.5 - tf.exp(conv_raw_dx1dy1)) * stride
        pred_xymax = (xy_grid + 0.5 + tf.exp(conv_raw_dx2dy2)) * stride
        pred_corner = tf.concat([pred_xymin, pred_xymax], axis=-1)
        # (2)对confidence进行decode
        pred_conf = tf.sigmoid(conv_raw_conf)

        # (3)对probability进行decode
        pred_prob = tf.sigmoid(conv_raw_prob)

        pred_bbox = tf.concat([pred_corner, pred_conf, pred_prob], axis=-1)
        return pred_bbox

    def forward_include_res_with_prune_factor_shortcut_list(self, inputs, prune_factor, shortcut_list,
                                                            is_training=False, reuse=False, prune_cnt=1):
        # the input img_size, form: [height, weight]
        # it will be used later
        self.img_size = tf.shape(inputs)[1:3]
        self.shortcut_list = shortcut_list
        # set batch norm params
        batch_norm_params = {
            'decay': self.batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': is_training,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(self.weight_decay)):
                with tf.variable_scope('darknet53_body'):
                    route_1, route_2, route_3 = parse_include_res_darknet53_body_prune_factor_shortcut_list(inputs,
                                                                                                            prune_factor,
                                                                                                            self.shortcut_list,
                                                                                                            prune_cnt)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block_pecentage(route_3, 512, prune_factor, prune_cnt=prune_cnt)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1, tf.shape(route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block_pecentage(concat1, 256, prune_factor, prune_cnt=prune_cnt)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2, tf.shape(route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block_pecentage(concat2, 128, prune_factor, prune_cnt=prune_cnt)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3


    def reorg_layer(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map)[1:3]  # [13, 13]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        box_centers = box_centers * ratio[::-1]

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
        # rescale to the original image scale
        box_sizes = box_sizes * ratio[::-1]

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def reorg_layer_no_scale(self, feature_map, anchors):
        '''
        feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
            from `forward` function
        anchors: shape: [3, 2]
        '''
        # NOTE: size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map)[1:3]  # [13, 13]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # rescale the anchors to the feature_map
        # NOTE: the anchor is in [w, h] format!
        rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

        feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + self.class_num])

        # split the feature_map along the last dimension
        # shape info: take 416x416 input image and the 13*13 feature_map for example:
        # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
        # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, self.class_num], axis=-1)
        box_centers = tf.nn.sigmoid(box_centers)

        # use some broadcast tricks to get the mesh coordinates
        grid_x = tf.range(grid_size[1], dtype=tf.int32)
        grid_y = tf.range(grid_size[0], dtype=tf.int32)
        grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
        x_offset = tf.reshape(grid_x, (-1, 1))
        y_offset = tf.reshape(grid_y, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        # shape: [13, 13, 1, 2]
        x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

        # get the absolute box coordinates on the feature_map
        box_centers = box_centers + x_y_offset
        # rescale to the original image scale
        # box_centers = box_centers * ratio[::-1]
        box_centers = box_centers * ratio

        # avoid getting possible nan value with tf.clip_by_value
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        # box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 100) * rescaled_anchors
        # rescale to the original image scale
        # box_sizes = box_sizes * ratio[::-1]
        box_sizes = box_sizes * ratio

        # shape: [N, 13, 13, 3, 4]
        # last dimension: (center_x, center_y, w, h)
        boxes = tf.concat([box_centers, box_sizes], axis=-1)

        # shape:
        # x_y_offset: [13, 13, 1, 2]
        # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
        # conf_logits: [N, 13, 13, 3, 1]
        # prob_logits: [N, 13, 13, 3, class_num]
        return x_y_offset, boxes, conf_logits, prob_logits

    def predict_no_stride(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer_no_scale(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        # center_x = boxes[:,:,0:1]
        # center_y = boxes[:,:,1:2]
        # width = boxes[:,:,2:3]
        # height = boxes[:,:,3:4]

        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs

    def predict(self, feature_maps):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps

        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs

    def predict2(self, feature_maps, img_size):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        feature_map_1, feature_map_2, feature_map_3 = feature_maps
        self.img_size = img_size
        feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
                               (feature_map_2, self.anchors[3:6]),
                               (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer(feature_map, anchors) for (feature_map, anchors) in feature_map_anchors]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

        return boxes, confs, probs

    def predict_single_featuremap(self, feature_map, img_size, feature_map_anchor):
        '''
        Receive the returned feature_maps from `forward` function,
        the produce the output predictions at the test stage.
        '''
        # feature_map_1, feature_map_2, feature_map_3 = feature_map
        self.img_size = img_size
        # feature_map_anchors = [(feature_map_1, self.anchors[6:9]),
        #                        (feature_map_2, self.anchors[3:6]),
        #                        (feature_map_3, self.anchors[0:3])]
        reorg_results = [self.reorg_layer_no_scale(feature_map, feature_map_anchor)]

        def _reshape(result):
            x_y_offset, boxes, conf_logits, prob_logits = result
            grid_size = tf.shape(x_y_offset)[:2]
            boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
            prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, self.class_num])
            # shape: (take 416*416 input image and feature_map_1 for example)
            # boxes: [N, 13*13*3, 4]
            # conf_logits: [N, 13*13*3, 1]
            # prob_logits: [N, 13*13*3, class_num]
            return boxes, conf_logits, prob_logits

        boxes_list, confs_list, probs_list = [], [], []
        for result in reorg_results:
            boxes, conf_logits, prob_logits = _reshape(result)
            confs = tf.sigmoid(conf_logits)
            probs = tf.sigmoid(prob_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            probs_list.append(probs)

        # collect results on three scales
        # take 416*416 input image for example:
        # shape: [N, (13*13+26*26+52*52)*3, 4]
        boxes = tf.concat(boxes_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, 1]
        confs = tf.concat(confs_list, axis=1)
        # shape: [N, (13*13+26*26+52*52)*3, class_num]
        probs = tf.concat(probs_list, axis=1)

        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2

        boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)
        pred_bbox = tf.concat([boxes, confs, probs], axis=-1)

        return pred_bbox
        # return boxes, confs, probs


    def loss_layer(self, feature_map_i, y_true, anchors):
        '''
        calc loss function from a certain scale
        input:
            feature_map_i: feature maps of a certain scale. shape: [N, 13, 13, 3*(5 + num_class)] etc.
            y_true: y_ture from a certain scale. shape: [N, 13, 13, 3, 5 + num_class + 1] etc.
            anchors: shape [9, 2]
        '''

        # size in [h, w] format! don't get messed up!
        grid_size = tf.shape(feature_map_i)[1:3]
        # the downscale ratio in height and weight
        ratio = tf.cast(self.img_size / grid_size, tf.float32)
        # N: batch_size
        N = tf.cast(tf.shape(feature_map_i)[0], tf.float32)

        x_y_offset, pred_boxes, pred_conf_logits, pred_prob_logits = self.reorg_layer(feature_map_i, anchors)

        ###########
        # get mask
        ###########
        # shape: take 416x416 input image and 13*13 feature_map for example:
        # [N, 13, 13, 3, 1]
        object_mask = y_true[..., 4:5]
        # shape: [N, 13, 13, 3, 4] & [N, 13, 13, 3] ==> [V, 4]
        # V: num of true gt box
        valid_true_boxes = tf.boolean_mask(y_true[..., 0:4], tf.cast(object_mask[..., 0], 'bool'))

        # shape: [V, 2]
        valid_true_box_xy = valid_true_boxes[:, 0:2]
        valid_true_box_wh = valid_true_boxes[:, 2:4]
        # shape: [N, 13, 13, 3, 2]
        pred_box_xy = pred_boxes[..., 0:2]
        pred_box_wh = pred_boxes[..., 2:4]

        # calc iou
        # shape: [N, 13, 13, 3, V]
        iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)

        # shape: [N, 13, 13, 3]
        best_iou = tf.reduce_max(iou, axis=-1)

        # get_ignore_mask
        ignore_mask = tf.cast(best_iou < 0.5, tf.float32)
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)

        # get xy coordinates in one cell from the feature_map
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_xy = y_true[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset

        # get_tw_th
        # numerical range: 0 ~ 1
        # shape: [N, 13, 13, 3, 2]
        true_tw_th = y_true[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # for numerical stability
        true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0),
                              x=tf.ones_like(true_tw_th), y=true_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0),
                              x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

        # box size punishment:
        # box with smaller area has bigger weight. This is taken from the yolo darknet C source code.
        # shape: [N, 13, 13, 3, 1]
        box_loss_scale = 2. - (y_true[..., 2:3] / tf.cast(self.img_size[1], tf.float32)) * (
                    y_true[..., 3:4] / tf.cast(self.img_size[0], tf.float32))

        ############
        # loss_part
        ############
        # mix_up weight
        # [N, 13, 13, 3, 1]
        mix_w = y_true[..., -1:]
        # shape: [N, 13, 13, 3, 1]
        xy_loss = tf.reduce_sum(tf.square(true_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(true_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                logits=pred_conf_logits)
        # TODO: may need to balance the pos-neg by multiplying some weights
        conf_loss = conf_loss_pos + conf_loss_neg
        if self.use_focal_loss:
            alpha = 1.0
            gamma = 2.0
            # TODO: alpha should be a mask array if needed
            focal_mask = alpha * tf.pow(tf.abs(object_mask - tf.sigmoid(pred_conf_logits)), gamma)
            conf_loss *= focal_mask
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N

        # shape: [N, 13, 13, 3, 1]
        # whether to use label smooth
        if self.use_label_smooth:
            delta = 0.01
            label_target = (1 - delta) * y_true[..., 5:-1] + delta * 1. / self.class_num
        else:
            label_target = y_true[..., 5:-1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target,
                                                                           logits=pred_prob_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N

        return xy_loss, wh_loss, conf_loss, class_loss

    def compute_loss(self, y_pred, y_true):
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]

    def compute_loss_knowledge_ditstill(self, teacher_output, y_pred, y_true, batchsize, num_classes):
        '''
        param:
            y_pred: returned feature_map list by `forward` function: [feature_map_1, feature_map_2, feature_map_3]
            y_true: input y_true by the tf.data pipeline
        '''
        loss_xy, loss_wh, loss_conf, loss_class = 0., 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        # calc loss in 3 scales
        for i in range(len(y_pred)):
            result = self.loss_layer(y_pred[i], y_true[i], anchor_group[i])
            loss_xy += result[0]
            loss_wh += result[1]
            loss_conf += result[2]
            loss_class += result[3]
        kd_loss = self.distillation_loss1(teacher_output, y_pred, num_classes=num_classes,  batch_size=batchsize)
        total_loss = loss_xy + loss_wh + loss_conf + loss_class + kd_loss
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class, kd_loss]

    def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        '''
        maintain an efficient way to calculate the ios matrix between ground truth true boxes and the predicted boxes
        note: here we only care about the size match
        '''
        # shape:
        # true_box_??: [V, 2]
        # pred_box_??: [N, 13, 13, 3, 2]

        # shape: [N, 13, 13, 3, 1, 2]
        pred_box_xy = tf.expand_dims(pred_box_xy, -2)
        pred_box_wh = tf.expand_dims(pred_box_wh, -2)

        # shape: [1, V, 2]
        true_box_xy = tf.expand_dims(true_box_xy, 0)
        true_box_wh = tf.expand_dims(true_box_wh, 0)

        # [N, 13, 13, 3, 1, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2]
        intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / 2.,
                                    true_box_xy - true_box_wh / 2.)
        intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,
                                    true_box_xy + true_box_wh / 2.)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)

        # shape: [N, 13, 13, 3, V]
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # shape: [N, 13, 13, 3, 1]
        pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]
        # shape: [1, V]
        true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]

        # [N, 13, 13, 3, V]
        iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

        return iou

    def distillation_loss1(self, output_s, output_t, num_classes, batch_size):
        batch_size = tf.cast(batch_size, tf.float32)
        T = 3.0
        Lambda_ST = 0.001
        # criterion_st = torch.nn.KLDivLoss(reduction='sum')
        output_s = tf.concat([tf.reshape(i, [-1, num_classes + 5]) for i in output_s], axis=-1)
        output_t = tf.concat([tf.reshape(i, [-1, num_classes + 5]) for i in output_t], axis=-1)
        T_prob = tf.nn.softmax(output_t/T, dim=1)
        S_prob = tf.nn.softmax(output_s, dim=1)

        loss_kn = tf.keras.losses.kld(T_prob,S_prob)/batch_size

        # loss_tf_kn = tf.reduce_sum(T_prob*tf.log(T_prob / S_prob), axis=-1)
        loss_torch_kn = tf.keras.losses.kld(T_prob,S_prob)/batch_size *(T*T) /batch_size *Lambda_ST
        # case_3 = tf.reduce_sum(p * tf.log(p) - p * tf.log(q)

        return loss_kn
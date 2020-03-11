import os
import numpy as np
import tensorflow as tf

import quant_yolo.utils as utils
from quant_yolo.config import cfg
from quant_yolo.darknet53 import darknet53_model
from quant_yolo.nn_skeleton import ModelSkeleton
import quant_yolo.args_quat as args

class yolov3_model(object):
    '''
    initial YOLO v3 structure definetion
    '''

    def __init__(self, input_data, trainable):
        self.nnlib = ModelSkeleton(trainable)

        self.trainable = trainable
        self.strides = np.array(cfg.COCO_STRIDES)
        self.anchors = utils.get_anchors(cfg.COCO_ANCHORS)
        self.classes = utils.read_class_names(args.class_name_path)
        # self.classes = utils.read_class_names(cfg.COCO_NAMES)
        self.num_class = len(self.classes)

        self.darknet53_model = darknet53_model(input_data, trainable)
        self.route1, self.route2, self.route3 = self.darknet53_model.route1, self.darknet53_model.route2, self.darknet53_model.route3
        self.lbbox, self.mbbox, self.sbbox = self._backbone()

        self.pred_sbbox = self.decode(self.sbbox, self.anchors[0], self.strides[0])
        self.pred_mbbox = self.decode(self.mbbox, self.anchors[1], self.strides[1])
        self.pred_lbbox = self.decode(self.lbbox, self.anchors[2], self.strides[2])

    def _backbone(self):
        # block_big_13x13
        blkb_conv1 = self.nnlib.conv_bn_leakyrelu_layer(self.route3, 'blkb_conv1_1s1', [1, 1, 1024, 512],
                                                        self.trainable)
        blkb_conv2 = self.nnlib.conv_bn_leakyrelu_layer(blkb_conv1, 'blkb_conv2_3s1', [3, 3, 512, 1024], self.trainable)
        blkb_conv3 = self.nnlib.conv_bn_leakyrelu_layer(blkb_conv2, 'blkb_conv3_1s1', [1, 1, 1024, 512], self.trainable)
        blkb_conv4 = self.nnlib.conv_bn_leakyrelu_layer(blkb_conv3, 'blkb_conv4_3s1', [3, 3, 512, 1024], self.trainable)
        blkb_branch = self.nnlib.conv_bn_leakyrelu_layer(blkb_conv4, 'blkb_conv5_1s1', [1, 1, 1024, 512],
                                                         self.trainable)

        blkb_conv6 = self.nnlib.conv_bn_leakyrelu_layer(blkb_branch, 'blkb_conv6_3s1', [3, 3, 512, 1024],
                                                        self.trainable)
        lbbox = self.nnlib.conv_layer(blkb_conv6, 'blkb_conv7_1s1p', [1, 1, 1024, 255], self.trainable)

        # block_middle_26x26
        blkm_conv1 = self.nnlib.conv_bn_leakyrelu_layer(blkb_branch, 'blkm_conv1_1s1uc', [1, 1, 512, 256],
                                                        self.trainable)
        upsample_data = self.nnlib.upsample(blkm_conv1)
        blkm_conv1 = tf.concat([self.route2, upsample_data], axis=-1)

        blkm_conv2 = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv1, 'blkm_conv2_1s1', [1, 1, 768, 256], self.trainable)
        blkm_conv3 = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv2, 'blkm_conv3_3s1', [3, 3, 256, 512], self.trainable)
        blkm_conv4 = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv3, 'blkm_conv4_1s1', [1, 1, 512, 256], self.trainable)
        blkm_conv5 = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv4, 'blkm_conv5_3s1', [3, 3, 256, 512], self.trainable)
        blkm_branch = self.nnlib.conv_bn_leakyrelu_layer(blkm_conv5, 'blkm_conv6_1s1', [1, 1, 512, 256], self.trainable)

        blkm_conv7 = self.nnlib.conv_bn_leakyrelu_layer(blkm_branch, 'blkm_conv7_3s1', [3, 3, 256, 512], self.trainable)
        mbbox = self.nnlib.conv_layer(blkm_conv7, 'blkm_conv8_1s1p', [1, 1, 512, 255], self.trainable)

        # block_small_52x52
        blks_conv1 = self.nnlib.conv_bn_leakyrelu_layer(blkm_branch, 'blks_conv1_1s1uc', [1, 1, 256, 128],
                                                        self.trainable)
        upsample_data = self.nnlib.upsample(blks_conv1)
        blks_conv1 = tf.concat([self.route1, upsample_data], axis=-1)

        blks_conv2 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv1, 'blks_conv2_1s1', [1, 1, 384, 128], self.trainable)
        blks_conv3 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv2, 'blks_conv3_3s1', [3, 3, 128, 256], self.trainable)
        blks_conv4 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv3, 'blks_conv4_1s1', [1, 1, 256, 128], self.trainable)
        blks_conv5 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv4, 'blks_conv5_3s1', [3, 3, 128, 256], self.trainable)
        blks_conv6 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv5, 'blks_conv6_1s1', [1, 1, 256, 128], self.trainable)

        blks_conv7 = self.nnlib.conv_bn_leakyrelu_layer(blks_conv6, 'blks_conv7_3s1', [3, 3, 128, 256], self.trainable)
        sbbox = self.nnlib.conv_layer(blks_conv7, 'blks_conv8_1s1p', [1, 1, 256, 255], self.trainable)

        return lbbox, mbbox, sbbox

    def decode(self, conv_output, anchors, stride):
        """
        return tensor of shape [batch_size, output_size, output_size, anchor_per_scale, 5 + num_classes]
               contains (x, y, w, h, score, probability)
        """
        conv_shape = tf.shape(conv_output)
        batch_size = conv_shape[0]
        width = conv_shape[1]
        height = conv_shape[2]
        anchor_per_scale = len(anchors)

        conv_output = tf.reshape(conv_output, (batch_size, width, height, anchor_per_scale, 5 + self.num_class))

        conv_raw_dxdy = conv_output[:, :, :, :, 0:2]
        conv_raw_dwdh = conv_output[:, :, :, :, 2:4]
        conv_raw_conf = conv_output[:, :, :, :, 4:5]
        conv_raw_prob = conv_output[:, :, :, :, 5:]

        y = tf.tile(tf.range(width, dtype=tf.int32)[:, tf.newaxis], [1, height])
        x = tf.tile(tf.range(height, dtype=tf.int32)[tf.newaxis, :], [width, 1])

        xy_grid = tf.concat([x[:, :, tf.newaxis], y[:, :, tf.newaxis]], axis=-1)
        xy_grid = tf.tile(xy_grid[tf.newaxis, :, :, tf.newaxis, :], [batch_size, 1, 1, anchor_per_scale, 1])
        xy_grid = tf.cast(xy_grid, tf.float32)

        pred_xy = (tf.sigmoid(conv_raw_dxdy) + xy_grid) * stride
        pred_wh = (tf.exp(conv_raw_dwdh) * anchors) * stride
        pred_xywh = tf.concat([pred_xy, pred_wh], axis=-1)

        pred_conf = tf.sigmoid(conv_raw_conf)
        pred_prob = tf.sigmoid(conv_raw_prob)

        return tf.concat([pred_xywh, pred_conf, pred_prob], axis=-1)

slim = tf.contrib.slim
class yolov3(object):

    def __init__(self, class_num, anchors, use_label_smooth=False, use_focal_loss=False, batch_norm_decay=0.999,
                 weight_decay=5e-4, use_static_shape=True):

        # self.anchors = [[10, 13], [16, 30], [33, 23],
        # [30, 61], [62, 45], [59,  119],
        # [116, 90], [156, 198], [373,326]]
        self.class_num = class_num
        self.anchors = anchors
        self.batch_norm_decay = batch_norm_decay
        self.use_label_smooth = use_label_smooth
        self.use_focal_loss = use_focal_loss
        self.weight_decay = weight_decay
        # inference speed optimization
        # if `use_static_shape` is True, use tensor.get_shape(), otherwise use tf.shape(tensor)
        # static_shape is slightly faster
        self.use_static_shape = use_static_shape

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
                    route_1, route_2, route_3 = darknet53_body(inputs)

                with tf.variable_scope('yolov3_head'):
                    inter1, net = yolo_block(route_3, 512)
                    feature_map_1 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_1 = tf.identity(feature_map_1, name='feature_map_1')

                    inter1 = conv2d(inter1, 256, 1)
                    inter1 = upsample_layer(inter1,
                                            route_2.get_shape().as_list() if self.use_static_shape else tf.shape(
                                                route_2))
                    concat1 = tf.concat([inter1, route_2], axis=3)

                    inter2, net = yolo_block(concat1, 256)
                    feature_map_2 = slim.conv2d(net, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_2 = tf.identity(feature_map_2, name='feature_map_2')

                    inter2 = conv2d(inter2, 128, 1)
                    inter2 = upsample_layer(inter2,
                                            route_1.get_shape().as_list() if self.use_static_shape else tf.shape(
                                                route_1))
                    concat2 = tf.concat([inter2, route_1], axis=3)

                    _, feature_map_3 = yolo_block(concat2, 128)
                    feature_map_3 = slim.conv2d(feature_map_3, 3 * (5 + self.class_num), 1,
                                                stride=1, normalizer_fn=None,
                                                activation_fn=None, biases_initializer=tf.zeros_initializer())
                    feature_map_3 = tf.identity(feature_map_3, name='feature_map_3')

            return feature_map_1, feature_map_2, feature_map_3

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


def darknet53_body(inputs):
    def res_block(inputs, filters):
        shortcut = inputs
        net = conv2d(inputs, filters * 1, 1)
        net = conv2d(net, filters * 2, 3)

        net = net + shortcut

        return net

    # first two conv2d layers
    net = conv2d(inputs, 32, 3, strides=1)
    net = conv2d(net, 64, 3, strides=2)

    # res_block * 1
    net = res_block(net, 32)

    net = conv2d(net, 128, 3, strides=2)

    # res_block * 2
    for i in range(2):
        net = res_block(net, 64)

    net = conv2d(net, 256, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 128)

    route_1 = net
    net = conv2d(net, 512, 3, strides=2)

    # res_block * 8
    for i in range(8):
        net = res_block(net, 256)

    route_2 = net
    net = conv2d(net, 1024, 3, strides=2)

    # res_block * 4
    for i in range(4):
        net = res_block(net, 512)
    route_3 = net

    return route_1, route_2, route_3

def yolo_block(inputs, filters):
    net = conv2d(inputs, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    net = conv2d(net, filters * 2, 3)
    net = conv2d(net, filters * 1, 1)
    route = net
    net = conv2d(net, filters * 2, 3)
    return route, net


def upsample_layer(inputs, out_shape):
    new_height, new_width = out_shape[1], out_shape[2]
    # NOTE: here height is the first
    # TODO: Do we need to set `align_corners` as True?
    # inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width), name='upsampled')
    inputs = tf.image.resize_bilinear(inputs, size=[new_height, new_width], name='upsampled')
    return inputs


def yolo_block_pecentage(inputs, filters, pecentage, prune_cnt=1):
    import numpy as np
    true_filters_1 = filters
    true_filters_2 = filters * 2
    for i in range(prune_cnt):
        true_filters_1 = np.floor(true_filters_1 * pecentage)
        true_filters_2 = np.floor(true_filters_2 * pecentage)

    # net = conv2d(inputs, np.floor(filters * 1 * pow(pecentage, prune_cnt)), 1)
    # net = conv2d(net,  np.floor(filters * 2 * pow(pecentage, prune_cnt)), 3)
    # net = conv2d(net,  np.floor(filters * 1 * pow(pecentage, prune_cnt)), 1)
    # net = conv2d(net,  np.floor(filters * 2 * pow(pecentage, prune_cnt)), 3)
    # net = conv2d(net, np.floor(filters * 1 * pow(pecentage, prune_cnt)), 1)
    # route = net
    # net = conv2d(net,  np.floor(filters * 2 * pow(pecentage, prune_cnt)), 3)

    net = conv2d(inputs, true_filters_1, 1)
    net = conv2d(net, true_filters_2, 3)
    net = conv2d(net, true_filters_1, 1)
    net = conv2d(net, true_filters_2, 3)
    net = conv2d(net, true_filters_1, 1)
    route = net
    net = conv2d(net, true_filters_2, 3)

    return route, net
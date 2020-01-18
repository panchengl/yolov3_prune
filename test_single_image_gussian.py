# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
import os
from model_sliming import sliming_yolov3
from gussian_yolo.model import yolov3
# from model import yolov3
os.environ['CUDA_VISIBLE_DEVICES']='1'
parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image", type=str, default="./img_dir",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/voc.names",
                    help="The path of the class names.")
# parser.add_argument("--restore_path", type=str, default="./sliming_checkpoint/sliming_prune_model_darknet_yolo_head.ckpt",
                    # help="The path of the weights to restore.")
# parser.add_argument("--restore_path", type=str, default="./sliming_checkpoint/best_model_Epoch_9_step_27669.0_mAP_0.5013_loss_8.8991_lr_4.8e-05",
#                     help="The path of the weights to restore.")
parser.add_argument("--restore_path", type=str, default="./gussian_yolo/checkpoint/best_model_Epoch_17_step_21293_mAP_0.8459_loss_0.5503_lr_0.0001",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    ##########################################################################################################
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    yolo_model = yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        pred_feature_maps = yolo_model.forward(input_data, False)
    pred_boxes, pred_confs, pred_probs, pred_sigma = yolo_model.predict(pred_feature_maps)
    ##########################################################################################################

    # sliming_yolo_model = yolov3(args.num_class, args.anchors)
    # with tf.variable_scope('yolov3'):
    #     pred_feature_maps = sliming_yolo_model.forward(input_data, prune_factor=0.8)
    # pred_boxes, pred_confs, pred_probs, pred_sigma = sliming_yolo_model.predict(pred_feature_maps)
    #
    print("sigma is",pred_sigma)
    print(" box is",pred_boxes)
    mena_sigma = tf.reduce_mean(pred_sigma, axis=-1, keepdims=True)
    print("mean_sigma is", mena_sigma)
    print("pred_confs is", pred_confs)
    print("pred_probs is", pred_probs)
    pred_scores = pred_confs * pred_probs*(1- mena_sigma)
    # pred_scores = pred_confs * pred_probs

    boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.4, nms_thresh=0.5)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)

    file_list = os.listdir(args.input_image)
    file_dir = '/home/pcl/tf_work/my_github/yolov3_prune/img_dir/'
    for img_name in file_list:
        img_ori = cv2.imread(file_dir + img_name)
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        # rescale the coordinates to the original image
        boxes_[:, 0] *= (width_ori/float(args.new_size[0]))
        boxes_[:, 2] *= (width_ori/float(args.new_size[0]))
        boxes_[:, 1] *= (height_ori/float(args.new_size[1]))
        boxes_[:, 3] *= (height_ori/float(args.new_size[1]))

        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)

        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]], color=color_table[labels_[i]])
        # cv2.imshow('Detection result', img_ori)
        cv2.imwrite('/home/pcl/tf_work/my_github/yolov3_prune/results/'+img_name, img_ori)
        cv2.waitKey(0)

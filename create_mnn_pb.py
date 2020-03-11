# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import os
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
import datetime
from sliming_yolo.model_sliming import sliming_yolov3
from model import yolov3
# from pruning_model import parse_darknet53_body
# from pruning_model import sparse_yolov3
# from model_sliming import sliming_yolov3

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("--input_image_dir", type=str,default = "/home/pcl/tf_work/YOLOv3_TensorFlow//img_dir",
                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="/home/pcl/tf_work/YOLOv3_TensorFlow//data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[608, 608],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="/home/pcl/tf_work/YOLOv3_TensorFlow//data/my_data/cheliang.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="/home/pcl/tf_work/YOLOv3_TensorFlow/dianli_608/fiveth_kmeans_checkpoint/best_model_Epoch_3_step_17895_mAP_0.5809_loss_3.4707_lr_0.0001",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)
# with tf.device('/cpu:0'):

def postprocess_doctor_yang(pred_bbox, test_input_size, org_img_shape):
    from utils.nms_utils import nms
    conf_thres = 0.3
    pred_bbox = np.array(pred_bbox)
    pred_coor = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]
    print(np.shape(pred_bbox))
    print(np.shape(pred_coor))
    print(np.shape(pred_conf))
    print(np.shape(pred_prob))
    org_h, org_w = org_img_shape
    print(org_h, org_w)
    # resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    resize_ratio = (1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    dw = (test_input_size - resize_ratio[0] * org_w) / 2
    dh = (test_input_size - resize_ratio[1] * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio[0]
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio[1]

    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]), np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    valid_scale = (0, np.inf)
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > conf_thres
    mask = np.logical_and(scale_mask, score_mask)
    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]
    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)
    bboxes = nms(bboxes, conf_thres, 0.45, method='nms')
    return bboxes

with tf.Session() as sess:
    input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
    # '''
    # frozen feature map pb model for yolov3, but pb model exclude post process, nms process, because of huawei a200 required post process must added in pb model,so i create another pb model
    # '''
    ################################### orignal model #######################################
    # yolo_model = yolov3(args.num_class, args.anchors)
    # with tf.variable_scope('yolov3'):
    #     pred_feature_maps = yolo_model.forward(input_data, False)
    # pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    #
    # pred_scores = pred_confs * pred_probs
    #
    # boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.3, nms_thresh=0.5)
    #
    # saver = tf.train.Saver()
    # saver.restore(sess, args.restore_path)
    # constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["yolov3/yolov3_head/feature_map_1", "yolov3/yolov3_head/feature_map_2", "yolov3/yolov3_head/feature_map_3"])
    # with tf.gfile.FastGFile("./dianli_tezhongcheliang_608_20200109.pb", mode='wb') as f:
    #     f.write(constant_graph.SerializeToString())
    #########################################################################################

    # yolo_model = sliming_yolov3(args.num_class, args.anchors)
    # with tf.variable_scope('yolov3'):
    #     pred_feature_maps = yolo_model.forward_include_res_with_prune_factor(input_data, 0.8, prune_cnt=5)
    # pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)
    # pred_scores = pred_confs * pred_probs
    # boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=30, score_thresh=0.3, nms_thresh=0.5)


    # yolo_model = sliming_yolov3(args.num_class, args.anchors)
    # with tf.variable_scope('yolov3'):
    #     # pred_bbox = yolo_model.forward_include_res_with_prune_factor_get_result_no_nms( input_data, 0.8, prune_cnt=5)
    #     pred_boxe, pred_conf, pred_prob = yolo_model.forward_include_res_with_prune_factor_docktor_yang( input_data, 0.8, prune_cnt=5)
    #     pred_scores = pred_conf * pred_prob
    #     boxes, scores, labels = gpu_nms(pred_boxe, pred_scores, args.num_class, max_boxes=30, score_thresh=0.3,
    #                                     nms_thresh=0.5)

    yolo_model = sliming_yolov3(args.num_class, args.anchors)
    with tf.variable_scope('yolov3'):
        boxes = yolo_model.forward_include_res_with_prune_factor_docktor_yang( input_data, 0.8, prune_cnt=5)

    saver = tf.train.Saver()
    saver.restore(sess, args.restore_path)
    constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["yolov3/yolov3_head/pred_boxes_last"])
    with tf.gfile.FastGFile("/home/pcl/tf_work/YOLOv3_TensorFlow/dianli_608/pb_model/dianli_608_fiveth_prune_20200310_doctor_yang_no_stride_07.pb", mode='wb') as f:
        f.write(constant_graph.SerializeToString())
    ############################################################################################
    img_list = os.listdir(args.input_image_dir)
    print(args.input_image_dir)
    for m in img_list:
        print(m)
        img_dir = os.path.join(args.input_image_dir, m)
        img_ori = cv2.imread(img_dir)
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.
        starttime = datetime.datetime.now()
        boxes_ = sess.run(boxes, feed_dict={input_data: img})
        print(np.shape(boxes_))
        endtime = datetime.datetime.now()
        last_result = postprocess_doctor_yang(boxes_, 608, img_ori.shape[:2])
        print("sess cost time is ", endtime - starttime)
        print("last result is ", last_result)
        print("box coords:")
        print('*' * 30)
        for i, box in enumerate(last_result):
            print(box)
            x0, y0, x1, y1, score, label  = box
            plot_one_box(img_ori, [x0, y0, x1, y1],  label=args.classes[int(label)] + ', {:.2f}%'.format(score * 100),
                         color=color_table[int(label)])
        cv2.namedWindow('Detection result', 0)
        cv2.imshow('Detection result', img_ori)
        # cv2.imwrite('/home/pcl/tf_work/YOLOv3_TensorFlow/results/' + m, img_ori)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

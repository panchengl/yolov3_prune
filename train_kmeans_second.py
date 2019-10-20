# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange

import args

from utils.data_utils import get_batch_data
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms
# from pruning_kneans_yolov3 import BasePruning
import os
from model_sliming import sliming_yolov3
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# setting loggers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S', filename=args.progress_log_path, filemode='w')

# setting placeholders
is_training = tf.placeholder(tf.bool, name="phase_train")
handle_flag = tf.placeholder(tf.string, [], name='iterator_handle_flag')
# register the gpu nms operation here for the following evaluation scheme
pred_boxes_flag = tf.placeholder(tf.float32, [1, None, None])
pred_scores_flag = tf.placeholder(tf.float32, [1, None, None])
gpu_nms_op = gpu_nms(pred_boxes_flag, pred_scores_flag, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)

##################
# tf.data pipeline
##################
train_dataset = tf.data.TextLineDataset(args.train_file)
train_dataset = train_dataset.shuffle(args.train_img_cnt)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'train', args.multi_scale_train, args.use_mix_up],
                         Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
train_dataset = train_dataset.prefetch(args.prefetech_buffer)

val_dataset = tf.data.TextLineDataset(args.val_file)
val_dataset = val_dataset.batch(1)
val_dataset = val_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'val', False, False],
                         Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
val_dataset.prefetch(args.prefetech_buffer)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
train_init_op = iterator.make_initializer(train_dataset)
val_init_op = iterator.make_initializer(val_dataset)

# get an element from the chosen dataset iterator
image_ids, image, y_true_13, y_true_26, y_true_52 = iterator.get_next()
y_true = [y_true_13, y_true_26, y_true_52]

# tf.data pipeline will lose the data `static` shape, so we need to set it manually
image_ids.set_shape([None])
image.set_shape([None, None, None, 3])
for y in y_true:
    y.set_shape([None, None, None, None, None])
##################
# Model prun
##################
# prune_iteration = 0
# prune_model = BasePruning(args.prune_factor, nb_finetune_epochs=args.nb_finetune_epochs,
#                           maximum_pruning_percent=args.maximum_pruning_percent,
#                           maximum_prune_iterations=args.maximum_prune_iterations,
#                           nb_trained_for_epochs=args.nb_trained_for_epochs,
#                           checkpoint_dir='/home/lovepan/work/YOLOv3_TensorFlow/checkpoint/')

# print("[INFO] model begining prune ...")
# prune_model.run_pruning()
# prune_iteration += 1
# print("[INFO] prune network completed")

####   reconstruction yolov3 net   ############
print('[INFO] begining construction yolov3 .....')
# prune_check_point_path = os.path.join(prune_model._checkpoint_dir, '200_epochs_kmeans_prune_restore_model_all.ckpt')
# input_data = tf.placeholder(tf.float32, [1, 416, 416, 3], name='input_data')
# yolo_prun_model = sparse_yolov3(args.class_num, args.anchors)
# with tf.variable_scope('yolov3'):
#     pred_feature_maps_fintune = yolo_prun_model.forward_include_res_with_prune_factor(image,
#                                                                                       prune_factor=prune_model._pruning_factor,
#                                                                                       is_training=is_training)
prune_model = sliming_yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay, args.weight_decay)
with tf.variable_scope('yolov3'):
    pred_feature_maps_fintune = prune_model.forward_include_res_with_prune_factor(image, prune_factor=0.8)
print("[INFO] prune network construction completed")

loss_fintune = prune_model.compute_loss(pred_feature_maps_fintune, y_true)
print("[INFO] loss calc completed")
y_pred_fintune = prune_model.predict(pred_feature_maps_fintune)
print("[INFO] prune network predict completed")
l2_loss_fintune = tf.losses.get_regularization_loss()
print("[INFO] l2 regularization loss calc completed")
saver_to_restore_finetune = tf.train.Saver(
    var_list=tf.contrib.framework.get_variables_to_restore(include= ['yolov3/darknet53_body','yolov3/yolov3_head']))

# prun_graph = tf.get_default_graph()  # 获得默认的图
tf.summary.scalar('train_batch_statistics/total_loss', loss_fintune[0])
tf.summary.scalar('train_batch_statistics/loss_xy', loss_fintune[1])
tf.summary.scalar('train_batch_statistics/loss_wh', loss_fintune[2])
tf.summary.scalar('train_batch_statistics/loss_conf', loss_fintune[3])
tf.summary.scalar('train_batch_statistics/loss_class', loss_fintune[4])
tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss_fintune)
tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss_fintune / loss_fintune[0])

global_step_finetune = tf.Variable(float(args.global_step), trainable=False,
                                   collections=[tf.GraphKeys.LOCAL_VARIABLES])
if args.use_warm_up:
    learning_rate_finetune = tf.cond(tf.less(global_step_finetune, args.train_batch_num * args.warm_up_epoch),
                                     lambda: args.learning_rate_init * global_step_finetune / (
                                             args.train_batch_num * args.warm_up_epoch),
                                     lambda: config_learning_rate(args,
                                                                  global_step_finetune - args.train_batch_num * args.warm_up_epoch))
else:
    learning_rate_finetune = config_learning_rate(args, global_step_finetune)
tf.summary.scalar('learning_rate', learning_rate_finetune)

if not args.save_optimizer:
    saver_best_finetune = tf.train.Saver()
    print("[INFO] save_best_finetune construct _not ")

optimizer_finetune = config_optimizer(args.optimizer_name, learning_rate_finetune)
update_vars_finetune = tf.contrib.framework.get_variables_to_restore(include= ['yolov3/darknet53_body','yolov3/yolov3_head'])
if args.save_optimizer:
    saver_best_finetune = tf.train.Saver()
    print("[INFO] save_best_finetune construct _true")

# set dependencies for BN ops
update_ops_finetune = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
grads_list = []
with tf.control_dependencies(update_ops_finetune):
    grads_varlist = optimizer_finetune.compute_gradients(loss_fintune[0] + l2_loss_fintune, var_list=update_vars_finetune)
    for i, (g, v) in enumerate(grads_varlist):
        if g is not None:
            grads_varlist[i] = (tf.clip_by_norm(g, 100), v)
    # # grad = tf.gradients(loss_fintune[0], update_vars_finetune)
    train_op_finetune = optimizer_finetune.apply_gradients(grads_varlist, global_step=global_step_finetune)
    # train_op_finetune = optimizer_finetune.minimize(loss_fintune[0] + l2_loss_fintune, var_list=update_vars_finetune,
    #                                                 global_step=global_step_finetune)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
#################################### begining finetune ###########################################
    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    print("[INFO] params completed initialized")

    # saver_to_restore_finetune.restore(sess, os.path.join(prune_model._checkpoint_dir, 'prue_channel_model.ckpt'))
    saver_to_restore_finetune.restore(sess, './kmeans_checkpoint/kmeans_best_model_Epoch_15_step_44271.0_mAP_0.5412_loss_9.6003_lr_4.423679e-05')
    # saver_to_restore_finetune.save(sess, os.path.join(prune_model._checkpoint_dir,
    #                                                   'kmeans_prune_restore_model_train.ckpt'))

    print("[INFO] model params restored")

    # print("[INFO] not finetine model params saved ")

    merged_fintune = tf.summary.merge_all()
    writer_fintune = tf.summary.FileWriter(args.log_dir, sess.graph)


    print('\n----------- start to finetune -----------\n')

    best_mAP_finetune = -np.Inf
    for epoch in range(args.total_epoches):

        sess.run(train_init_op)
        loss_total_finetune, loss_xy_finetune, loss_wh_finetune, loss_conf_finetune, loss_class_finetune = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for i in trange(args.train_batch_num):
            _, summary_finetune, __y_pred_finetune, __y_true_finetune, __loss_finetune, __global_step_finetune, __lr_finetune,= sess.run(
                [train_op_finetune, merged_fintune, y_pred_fintune, y_true, loss_fintune, global_step_finetune, learning_rate_finetune],
                feed_dict={is_training: True})
            # try:
            #     __grads = sess.run(grads_list, feed_dict={is_training: True})
            #     # for i in __grads:
            #     #     print('current l2 is ', np.linalg.norm(i))
            # except:
            #     print('grads clac is error')
            writer_fintune.add_summary(summary_finetune, global_step=__global_step_finetune)
            loss_total_finetune.update(__loss_finetune[0], len(__y_true_finetune[0]))
            loss_xy_finetune.update(__loss_finetune[1], len(__y_true_finetune[0]))
            loss_wh_finetune.update(__loss_finetune[2], len(__y_true_finetune[0]))
            loss_conf_finetune.update(__loss_finetune[3], len(__y_true_finetune[0]))
            loss_class_finetune.update(__loss_finetune[4], len(__y_true_finetune[0]))

            if __global_step_finetune % args.train_evaluation_step == 0 and __global_step_finetune > 0:
                # recall, precision = evaluate_on_cpu(__y_pred, __y_true, args.class_num, args.nms_topk, args.score_threshold, args.eval_threshold)
                recall_finetune, precision_finetune = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __y_pred_finetune,
                                                    __y_true_finetune, args.class_num, args.eval_threshold)

                info_finetune = "Epoch: {}, global_step: {} | loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f} | ".format(
                    epoch, int(__global_step_finetune), loss_total_finetune.avg, loss_xy_finetune.avg, loss_wh_finetune.avg, loss_conf_finetune.avg, loss_class_finetune.avg)
                info_finetune += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall_finetune, precision_finetune, __lr_finetune)
                print(info_finetune)
                logging.info(info_finetune)

                writer_fintune.add_summary(make_summary('evaluation/train_batch_recall', recall_finetune), global_step=__global_step_finetune)
                writer_fintune.add_summary(make_summary('evaluation/train_batch_precision', precision_finetune),
                                   global_step=__global_step_finetune)

                if np.isnan(loss_total_finetune.last_avg):
                    print('****' * 10)
                    raise ArithmeticError(
                        'Gradient exploded! Please train again and you may need modify some parameters.')

        if epoch % args.val_evaluation_epoch == 0 and epoch > 0:
            sess.run(val_init_op)

            val_loss_total_finetune, val_loss_xy_finetune, val_loss_wh_finetune, val_loss_conf_finetune, val_loss_class_finetune = \
                AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

            val_preds_finetune = []

            for j in trange(args.val_img_cnt):
                __image_ids_finetune, __y_pred_finetune, __loss_finetune = sess.run([image_ids, y_pred_fintune, loss_fintune],
                                                         feed_dict={is_training: False})
                pred_content_finetune = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids_finetune, __y_pred_finetune)
                val_preds_finetune.extend(pred_content_finetune)
                val_loss_total_finetune.update(__loss_finetune[0])
                val_loss_xy_finetune.update(__loss_finetune[1])
                val_loss_wh_finetune.update(__loss_finetune[2])
                val_loss_conf_finetune.update(__loss_finetune[3])
                val_loss_class_finetune.update(__loss_finetune[4])

            # calc mAP
            rec_total_finetune, prec_total_finetune, ap_total_finetune = AverageMeter(), AverageMeter(), AverageMeter()
            gt_dict_finetune = parse_gt_rec(args.val_file, args.img_size)

            info_finetune = '======> Epoch: {}, global_step: {}, lr: {:.6g} <======\n'.format(epoch, __global_step_finetune, __lr_finetune)
            preci_myself = 0
            obj_sum_finetune = 0
            right_obj_finetune = 0
            for ii in range(args.class_num):
                npos_finetune, nd_finetune, rec_finetune, prec_finetune, ap_finetune = voc_eval(gt_dict_finetune, val_preds_finetune, ii, iou_thres=args.eval_threshold,
                                                   use_07_metric=False)
                # print("pres is :{:0.4f}".format(prec))
                info_finetune += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec_finetune, prec_finetune, ap_finetune)
                rec_total_finetune.update(rec_finetune, npos_finetune)
                prec_total_finetune.update(prec_finetune, nd_finetune)
                obj_sum_finetune += nd_finetune
                right_obj_finetune += npos_finetune
                ap_total_finetune.update(ap_finetune, 1)
            pre_finetune = prec_total_finetune.avg
            mAP_finetune = ap_total_finetune.avg
            info_finetune += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total_finetune.avg, prec_total_finetune.avg, mAP_finetune)
            info_finetune += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n'.format(
                val_loss_total_finetune.avg, val_loss_xy_finetune.avg, val_loss_wh_finetune.avg, val_loss_conf_finetune.avg, val_loss_class_finetune.avg)
            print(info_finetune)
            logging.info(info_finetune)

            if mAP_finetune > best_mAP_finetune:
                best_mAP_finetune = mAP_finetune
                saver_best_finetune.save(sess,
                                './kmeans_checkpoint/' + 'kmeans_best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'.format(
                                    epoch, __global_step_finetune, best_mAP_finetune, val_loss_total_finetune.last_avg, __lr_finetune))

            writer_fintune.add_summary(make_summary('evaluation/val_mAP', mAP_finetune), global_step=epoch)
            writer_fintune.add_summary(make_summary('evaluation/val_recall', rec_total_finetune.last_avg), global_step=epoch)
            writer_fintune.add_summary(make_summary('evaluation/val_precision', prec_total_finetune.last_avg), global_step=epoch)
            writer_fintune.add_summary(make_summary('validation_statistics/total_loss', val_loss_total_finetune.last_avg),
                               global_step=epoch)
            writer_fintune.add_summary(make_summary('validation_statistics/loss_xy', val_loss_xy_finetune.last_avg), global_step=epoch)
            writer_fintune.add_summary(make_summary('validation_statistics/loss_wh', val_loss_wh_finetune.last_avg), global_step=epoch)
            writer_fintune.add_summary(make_summary('validation_statistics/loss_conf', val_loss_conf_finetune.last_avg),
                               global_step=epoch)
            writer_fintune.add_summary(make_summary('validation_statistics/loss_class', val_loss_class_finetune.last_avg),
                               global_step=epoch)
        # saver_best_fintune = tf.train.Saver()
        if epoch % 30 ==0  and epoch > 0:
            saver_best_finetune.save(sess, os.path.join('./kmeans_checkpoint/', ('kmeans_prune_restore_model_epoch_' + str(int(epoch)) + '_all.ckpt')))

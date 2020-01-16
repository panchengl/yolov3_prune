# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import logging
from tqdm import trange
import tensorflow.contrib.slim as slim
import os
# os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
# import args
from utils.data_utils import get_batch_data
from utils.misc_utils import shuffle_and_overwrite, make_summary, config_learning_rate, config_optimizer, AverageMeter
from utils.eval_utils import evaluate_on_cpu, evaluate_on_gpu, get_preds_gpu, voc_eval, parse_gt_rec
from utils.nms_utils import gpu_nms

from model import yolov3
import mutili_gpu_args as args
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def sum_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    sum_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_sum(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        sum_grads.append(grad_and_var)
    return sum_grads

def get_restorer():
    checkpoint_path = args.restore_path
    print("model restore from pretrained mode, path is :", checkpoint_path)

    model_variables = slim.get_model_variables()
    for var in model_variables:
        print(var.name)
    print(20*"__++__++__")

    def name_in_ckpt(var):
        return var.op.name

    nameInCkpt_Var_dict = {}
    for var in model_variables:
        if "Momentum" not in var.name:
            for exclude_var in args.restore_exclude:
                if exclude_var not in var.name:
                    var_name_in_ckpt = name_in_ckpt(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
    restore_variables = nameInCkpt_Var_dict
    for key, item in restore_variables.items():
        print("var_in_graph: ", item.name)
        print("var_in_ckpt: ", key)
        print(20*"___")
    restorer = tf.train.Saver(restore_variables)
    print(20 * "****")
    print("restore from pretrained_weighs in IMAGE_NET")
    return restorer, checkpoint_path

def print_info(image):
    # print("all batch img-ids is", image)
    return 0

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
########################  batchsize use 1 will cause gpu1 cannot allocate data, so 1 -> 2 ###########################
# if args.batch_size == 1:
#     args.batch_size == 2
train_dataset = tf.data.TextLineDataset(args.train_file)
train_dataset = train_dataset.shuffle(args.train_img_cnt)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'train', args.multi_scale_train, args.use_mix_up, args.letterbox_resize],
                         Tout=[tf.int64, tf.float32, tf.float32, tf.float32, tf.float32]),
    num_parallel_calls=args.num_threads
)
train_dataset = train_dataset.prefetch(args.prefetech_buffer)

val_dataset = tf.data.TextLineDataset(args.val_file)
val_dataset = val_dataset.batch(1)
# val_dataset = val_dataset.batch(args.batch_size)
val_dataset = val_dataset.map(
    lambda x: tf.py_func(get_batch_data,
                         inp=[x, args.class_num, args.img_size, args.anchors, 'val', False, False, args.letterbox_resize],
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

# img_name_batch, img_batch, gtboxes_and_label_batch, num_objects_batch, img_h_batch, img_w_batch = \
inputs_list = []
y_true_batch= []
image_ids_batch = []
image_batch = []
for i in range(args.NUM_GPU):
    start = i*(args.batch_size//args.NUM_GPU)
    end = (i+1)*(args.batch_size//args.NUM_GPU)

    img = image[start:end, :, :, :]
    id_img = image_ids[start:end]


    y_true_13_batch = y_true_13[start:end, :, :]
    y_true_26_batch = y_true_26[start:end, :, :]
    y_true_52_batch = y_true_52[start:end, :, :]
    imag_info = tf.py_func(print_info, inp=[y_true_13_batch], Tout=[tf.int64])

    y_true_batch.append([y_true_13_batch, y_true_26_batch, y_true_52_batch])
    image_batch.append(img)
    image_ids_batch.append(id_img)

    image_ids_batch[i].set_shape([None])
    image_batch[i].set_shape([None, None, None, 3])
    for y in y_true_batch[i]:
        y.set_shape([None, None, None, None, None])


# with tf.Session() as sess:
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     for epoch in range(args.total_epoches):
#         sess.run(train_init_op)
#         for i in trange(args.train_batch_num ):
#         # for i in trange(1):
#             a, b, c, d= sess.run(
#                 [image_ids_batch, image_batch, y_true_batch, imag_info] )
#             print('img_id id bantch is', a)
#             print('y_ture_batch is', y_true_batch)
#         print("one epoch is finished ")
#         print("one epoch is finished ")
#         print("one epoch is finished ")
#         print("one epoch is finished ")
            # print('e is', e)
            # print('gpu0 y_true batch is', c)
            # print('gpu1 y_true batch is', d)


with tf.device('/cpu:0'):
    tower_grads = []
    y_pred = []
    loss_gpus = []
    yolo_model = yolov3(args.class_num, args.anchors, args.use_label_smooth, args.use_focal_loss, args.batch_norm_decay, args.weight_decay, use_static_shape=False)
    global_step = tf.Variable(float(args.global_step), trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(args.NUM_GPU):
            print("current gpu is", i)
            with tf.device('/gpu:%d' % i):
                image_gpu_true = image_batch[i]
                y_gpu_true = y_true_batch[i]
                with tf.variable_scope('yolov3'):
                    pred_feature_maps = yolo_model.forward(image_gpu_true, is_training=is_training)
                    print("featuremap shape is", pred_feature_maps)
                loss = yolo_model.compute_loss(pred_feature_maps, y_gpu_true)
                y_pred_net = yolo_model.predict(pred_feature_maps)
                loss_gpus.append(loss)
                y_pred.append(y_pred_net)

                l2_loss = tf.losses.get_regularization_loss()  #[total_loss, loss_xy, loss_wh, loss_conf, loss_class]
                if args.use_warm_up:
                    learning_rate = tf.cond(tf.less(global_step, args.train_batch_num * args.warm_up_epoch),
                                            lambda: args.learning_rate_init * global_step / (
                                                        args.train_batch_num * args.warm_up_epoch),
                                            lambda: config_learning_rate(args,
                                                                         global_step - args.train_batch_num * args.warm_up_epoch))
                else:
                    learning_rate = config_learning_rate(args, global_step)
                tf.summary.scalar('learning_rate', learning_rate)
                if not args.save_optimizer:
                    saver_to_save = tf.train.Saver()
                    saver_best = tf.train.Saver()
                optimizer = config_optimizer(args.optimizer_name, learning_rate)
                saver_to_restore = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(include=args.restore_include, exclude=args.restore_exclude))
                update_vars = tf.contrib.framework.get_variables_to_restore(include=args.update_part)

                tf.summary.scalar('train_batch_statistics/total_loss', loss[0])
                tf.summary.scalar('train_batch_statistics/loss_xy', loss[1])
                tf.summary.scalar('train_batch_statistics/loss_wh', loss[2])
                tf.summary.scalar('train_batch_statistics/loss_conf', loss[3])
                tf.summary.scalar('train_batch_statistics/loss_class', loss[4])
                tf.summary.scalar('train_batch_statistics/loss_l2', l2_loss)
                tf.summary.scalar('train_batch_statistics/loss_ratio', l2_loss / loss[0])

                # set dependencies for BN ops
                total_losses = 0.0
                total_losses += loss[0]
                total_losses = total_losses / args.NUM_GPU
                if i == args.NUM_GPU - 1:
                    l2_loss = tf.losses.get_regularization_loss()
                    total_losses = total_losses + l2_loss
                # l2_loss = tf.losses.get_regularization_loss()
                # total_losses = total_losses + l2_loss
                tf.get_variable_scope().reuse_variables()
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    # gvs = optimizer.compute_gradients(loss[0] + l2_loss, var_list=update_vars)
                    # clip_grad_var = [gv if gv[0] is None else [
                    #     tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in gvs]
                    grads = optimizer.compute_gradients(total_losses)
                    clip_grad_var = [gv if gv[0] is None else [
                        tf.clip_by_norm(gv[0], 100.), gv[1]] for gv in grads]
                    tower_grads.append(clip_grad_var)

    if len(tower_grads) > 1:
        clip_grad_var = sum_gradients(tower_grads)
    else:
        clip_grad_var = tower_grads[0]

    train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

    if args.save_optimizer:
        print('Saving optimizer parameters to checkpoint! Remember to restore the global_step in the fine-tuning afterwards.')
        saver_to_save = tf.train.Saver()
        saver_best = tf.train.Saver()

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        saver_to_restore.restore(sess, args.restore_path)
        # restorer, restore_ckpt = get_restorer()
        # restorer.restore(sess, restore_ckpt)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

        print('\n----------- start to train -----------\n')
        best_mAP = -np.Inf

        for epoch in range(args.total_epoches):
            sess.run(train_init_op)
            loss_total, loss_xy, loss_wh, loss_conf, loss_class = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
            for i in trange(args.train_batch_num):
                if i <=  (args.train_batch_num-2):
                    _, summary, __y_pred, __y_true, __loss, __global_step, __lr = sess.run(
                        [train_op, merged, y_pred, y_true_batch, loss_gpus,  global_step, learning_rate],
                        feed_dict={is_training: True})
                    writer.add_summary(summary, global_step=__global_step)
                    loss_total.update((__loss[0][0] + __loss[1][0] ), len(__y_pred[0][0] + __y_pred[1][0]))
                    loss_xy.update((__loss[0][1] + __loss[1][1] ), len(__y_pred[0][0] + __y_pred[1][0]))
                    loss_wh.update((__loss[0][2] + __loss[1][2] ), len(__y_pred[0][0] + __y_pred[1][0]))
                    loss_conf.update((__loss[0][3] + __loss[1][3] ), len(__y_pred[0][0] + __y_pred[1][0]))
                    loss_class.update((__loss[0][4] + __loss[1][4] ), len(__y_pred[0][0] + __y_pred[1][0]))


                    if __global_step % args.train_evaluation_step == 0 and __global_step > 0:
                        print("loss total is", loss_total.average)
                        # recall, precision = evaluate_on_cpu(__y_pred, __y_true, args.class_num, args.nms_topk, args.score_threshold, args.nms_threshold)
                        recall, precision = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __y_pred[0], __y_true[0], args.class_num, args.nms_threshold)
                        recall_1, precision_1 = evaluate_on_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __y_pred[1], __y_true[1], args.class_num, args.nms_threshold)

                        recall_muti_gpu = (recall + recall_1)/2.0
                        precision_muti_gpu = (precision + precision_1)/2.0

                        print("precision_0, recall_0 is", precision, recall)
                        print("precision_1, recall_1 is", precision_1, recall_1)
                        info = "Epoch: {}, global_step: {} | loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f} | ".format(
                                epoch, int(__global_step), loss_total.average, loss_xy.average, loss_wh.average, loss_conf.average, loss_class.average)
                        info += 'Last batch: rec: {:.3f}, prec: {:.3f} | lr: {:.5g}'.format(recall_muti_gpu, precision_muti_gpu, __lr)
                        print(info)
                        logging.info(info)

                        writer.add_summary(make_summary('evaluation/train_batch_recall', recall_muti_gpu), global_step=__global_step)
                        writer.add_summary(make_summary('evaluation/train_batch_precision', precision_muti_gpu), global_step=__global_step)

                        if np.isnan(loss_total.average):
                            print('****' * 10)
                            raise ArithmeticError(
                                'Gradient exploded! Please train again and you may need modify some parameters.')

            # NOTE: this is just demo. You can set the conditions when to save the weights.
            if epoch % args.save_epoch == 0 and epoch > 0:
                if loss_total.average <= 2.:
                    saver_to_save.save(sess, args.save_dir + 'model-epoch_{}_step_{}_loss_{:.4f}_lr_{:.5g}'.format(epoch, int(__global_step), loss_total.average, __lr))

            # switch to validation dataset for evaluation
            if epoch % args.val_evaluation_epoch == 0 and epoch >= args.warm_up_epoch:
            # if epoch % args.val_evaluation_epoch == 0 :
                sess.run(val_init_op)
                val_loss_total, val_loss_xy, val_loss_wh, val_loss_conf, val_loss_class = \
                    AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

                val_preds = []

                for j in trange(args.val_img_cnt):
                    __image_ids, __y_pred, __loss = sess.run([image_ids_batch[0], y_pred[0], loss_gpus[0]],
                                                             feed_dict={is_training: False})

                    pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids, __y_pred)
                    # pred_content = get_preds_gpu(sess, gpu_nms_op, pred_boxes_flag, pred_scores_flag, __image_ids[0], __y_pred[0])
                    val_preds.extend(pred_content)
                    val_loss_total.update(__loss[0])
                    val_loss_xy.update(__loss[1])
                    val_loss_wh.update(__loss[2])
                    val_loss_conf.update(__loss[3])
                    val_loss_class.update(__loss[4])

                # calc mAP
                rec_total, prec_total, ap_total = AverageMeter(), AverageMeter(), AverageMeter()
                gt_dict = parse_gt_rec(args.val_file, args.img_size, args.letterbox_resize)

                info = '======> Epoch: {}, global_step: {}, lr: {:.6g} <======\n'.format(epoch, __global_step, __lr)

                for ii in range(args.class_num):
                    npos, nd, rec, prec, ap = voc_eval(gt_dict, val_preds, ii, iou_thres=args.eval_threshold, use_07_metric=args.use_voc_07_metric)
                    info += 'EVAL: Class {}: Recall: {:.4f}, Precision: {:.4f}, AP: {:.4f}\n'.format(ii, rec, prec, ap)
                    rec_total.update(rec, npos)
                    prec_total.update(prec, nd)
                    ap_total.update(ap, 1)

                mAP = ap_total.average
                info += 'EVAL: Recall: {:.4f}, Precison: {:.4f}, mAP: {:.4f}\n'.format(rec_total.average, prec_total.average, mAP)
                info += 'EVAL: loss: total: {:.2f}, xy: {:.2f}, wh: {:.2f}, conf: {:.2f}, class: {:.2f}\n'.format(
                    val_loss_total.average, val_loss_xy.average, val_loss_wh.average, val_loss_conf.average, val_loss_class.average)
                print(info)
                logging.info(info)

                if mAP > best_mAP:
                    best_mAP = mAP
                    saver_best.save(sess, args.save_dir + 'best_model_Epoch_{}_step_{}_mAP_{:.4f}_loss_{:.4f}_lr_{:.7g}'.format(
                                       epoch, int(__global_step), best_mAP, val_loss_total.average, __lr))

                writer.add_summary(make_summary('evaluation/val_mAP', mAP), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_recall', rec_total.average), global_step=epoch)
                writer.add_summary(make_summary('evaluation/val_precision', prec_total.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/total_loss', val_loss_total.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_xy', val_loss_xy.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_wh', val_loss_wh.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_conf', val_loss_conf.average), global_step=epoch)
                writer.add_summary(make_summary('validation_statistics/loss_class', val_loss_class.average), global_step=epoch)


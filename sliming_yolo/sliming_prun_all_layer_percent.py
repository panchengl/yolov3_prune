import abc
import re
import tempfile
import traceback
import os
from typing import Tuple, Callable, Union, List, Optional
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import models, layers
from sklearn import cluster, metrics
from utils.misc_utils import parse_anchors, read_class_names
from sliming_yolo.model_sliming import sliming_yolov3

# from ridurre import base_filter_pruning
root = '/home/pcl/tensorflow1.12/my_github/YOLOv3_TensorFlow'
anchor_path = root + '/data/yolo_anchors.txt'  # The path of the anchor txt file.
class_name_path = root + '/data/my_data/voc.names'  # The path of the class names.
anchors = parse_anchors(anchor_path)
num_class = len(read_class_names(class_name_path))

class SlimPruning:
    _FUZZ_EPSILON = 1e-5
    def __init__(self, pruning_factor: float, prune_iter_cnt: int, checkpoint_dir: str):
        self._pruning_factor = pruning_factor
        self._tmp_model_file_name = tempfile.NamedTemporaryFile().name
        self._channel_number_bins = None
        self._pruning_factors_for_channel_bins = None
        self._prune_iter_cnt = prune_iter_cnt
        self._original_number_of_filters = -1
        self._checkpoint_dir = checkpoint_dir

        # TODO: select a subset of layers to prune
        self._prunable_layers_regex = ".*"
        self._restore_part_first = ['yolov3/darknet53_body', 'yolov3/yolov3_head']
        self._restore_part_second = ['yolov3/darknet53_body','yolov3/yolov3_head']
        self._update_part = ['yolov3/yolov3_head']
        self._img_size = [416, 416]

    def run_pruning(self, prune_factor = 0.8, custom_objects_inside_model: dict = None) -> Tuple[models.Model, int]:
        pruning_iteration = 0
        while True:
            if prune_factor is not None:
                self._pruning_factor = prune_factor
                # Pruning step
                print("Running filter pruning {0}".format(pruning_iteration))
                weight_dict = self._prune_first_stage()
                sliming_yolo_model = self._reconstruction_model(weight_dict)
            print("Pruning stopped.")
            # return model, self._current_nb_of_epochs
            return sliming_yolo_model, weight_dict

    def _reconstruction_model(self, weight_dict):
        with tf.Session() as sess:
            input_data_2 = tf.placeholder(tf.float32, [1, self._img_size[1], self._img_size[0], 3], name='input_data')
            sliming_yolo_model = sliming_yolov3(num_class, anchors)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = sliming_yolo_model.forward_include_res_with_prune_factor(input_data_2,
                                                                                          prune_factor=self._pruning_factor,
                                                                                          is_training=True, prune_cnt=self._prune_iter_cnt)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            for layer_name, weight in weight_dict.items():
                print(layer_name)
                print(weight.shape)
                current_layer_tensor = tf.get_default_graph().get_tensor_by_name(layer_name)
                sess.run(tf.assign(current_layer_tensor, weight, validate_shape=True))
            print("reconstruction network completed")
            print("completed initialized")
            saver_best = tf.train.Saver()
            saver_best.save(sess, os.path.join('./sliming_checkpoint/', 'sliming_prune_model_darknet_yolo_head_fourth.ckpt'))
            return sliming_yolo_model

    def _run_pruning_for_conv2d_layer(self, pruning_factor: float, gamma_layer_weight_mtx):
        import operator
        _nb_channels = gamma_layer_weight_mtx.shape[0]
        abs_weight = list(map(abs, gamma_layer_weight_mtx))
        sorted_weight = np.sort(abs_weight)
        # print('curent is sorted_weight', sorted_weight)
        # print('thr id is ', np.ceil(  len(abs_weight) * (1 - pruning_factor )) )
        pruning_threshold = sorted_weight[  int(np.ceil(  len(abs_weight) * (1 - pruning_factor ) ) )]
        # print('current thr is ', pruning_threshold)
        bool_weight_indice = np.logical_not(abs_weight < pruning_threshold)
        save_indices = []
        for i, j in enumerate(bool_weight_indice):
            # print('current id is ', i)
            # print('current bool is ', j)
            if j == True:
                save_indices.append(i)
        # Compute filter indices which can be pruned
        channel_indices = set(np.arange(len(gamma_layer_weight_mtx)))
        channel_indices_to_keep = set(save_indices)
        channel_indices_to_prune = list(channel_indices.difference(channel_indices_to_keep))
        channel_indices_to_keep = list(channel_indices_to_keep)
        # print('prune ind is ', channel_indices_to_prune)
        nb_of_clusters = len(save_indices)
        saved_prune_channel = int(np.floor(self._pruning_factor * _nb_channels))
        print('true saved channel is ', nb_of_clusters)
        print('calc saved channel is ', saved_prune_channel)
        if len(channel_indices_to_keep) < saved_prune_channel:
            print("Number of selected channels for pruning is less than expected")
            bn_weights_key = dict()
            diff = saved_prune_channel - len(channel_indices_to_keep)
            print("Randomly adding {0} channels for pruning".format(diff))
            for i in range(len(channel_indices_to_prune)):
                bn_weights_key[channel_indices_to_prune[i]] = abs_weight[channel_indices_to_prune[i]]
            sorted(bn_weights_key.items(), key=lambda item: item[1], reverse=True)
            keys = list(bn_weights_key.keys())
            for j in range(diff):
                channel_indices_to_keep.append(keys[j])
                channel_indices_to_prune.pop(keys[j])
            # np.random.shuffle(channel_indices_to_keep)
            # for i in range(diff):
            #     channel_indices_to_prune.append(channel_indices_to_keep.pop(i))
        elif len(channel_indices_to_keep) > saved_prune_channel:
            print("Number of selected channels for pruning is greater than expected. Leaving too few channels.")
            diff = len(channel_indices_to_keep) - saved_prune_channel
            print("Discarding {0} pruneable channels".format(diff))
            bn_weights_key_2 = dict()
            for i in range(len(channel_indices_to_keep)):
                bn_weights_key_2[channel_indices_to_keep[i]] = abs_weight[channel_indices_to_keep[i]]
            sorted(bn_weights_key_2.items(), key=lambda item: item[1])
            keys = list(bn_weights_key_2.keys())
            for j in range(diff):
                channel_indices_to_prune.append(keys[j])
                channel_indices_to_keep.pop(keys[j])
            # for i in range(diff):
            #     channel_indices_to_keep.append(channel_indices_to_prune.pop(i))

        if len(channel_indices_to_keep) != saved_prune_channel:
            raise ValueError(
                "Number of clusters {0} is not equal with the selected "
                "pruneable channels {1}".format(nb_of_clusters, len(channel_indices_to_prune)))
        #####################################################################################
        return channel_indices_to_prune

    def _run_pruning_for_all_layer(self, pruning_threshold: float, gamma_layer_weight_mtx):
        import operator
        _nb_channels = gamma_layer_weight_mtx.shape[0]
        abs_weight = list(map(abs, gamma_layer_weight_mtx))
        # sorted_weight = np.sort(abs_weight)
        # # print('curent is sorted_weight', sorted_weight)
        # # print('thr id is ', np.ceil(  len(abs_weight) * (1 - pruning_factor )) )
        # pruning_threshold = sorted_weight[  int(np.ceil(  len(abs_weight) * (1 - pruning_factor ) ) )]
        # # print('current thr is ', pruning_threshold)
        bool_weight_indice = np.logical_not(abs_weight < pruning_threshold)
        save_indices = []
        for i, j in enumerate(bool_weight_indice):
            # print('current id is ', i)
            # print('current bool is ', j)
            if j == True:
                save_indices.append(i)
        # Compute filter indices which can be pruned
        channel_indices = set(np.arange(len(gamma_layer_weight_mtx)))
        channel_indices_to_keep = set(save_indices)
        channel_indices_to_prune = list(channel_indices.difference(channel_indices_to_keep))
        channel_indices_to_keep = list(channel_indices_to_keep)
        # print('prune ind is ', channel_indices_to_prune)
        # nb_of_clusters = len(save_indices)
        # saved_prune_channel = int(np.floor(self._pruning_factor * _nb_channels))
        # print('true saved channel is ', nb_of_clusters)
        # print('calc saved channel is ', saved_prune_channel)
        # if len(channel_indices_to_keep) < saved_prune_channel:
        #     print("Number of selected channels for pruning is less than expected")
        #     bn_weights_key = dict()
        #     diff = saved_prune_channel - len(channel_indices_to_keep)
        #     print("Randomly adding {0} channels for pruning".format(diff))
        #     for i in range(len(channel_indices_to_prune)):
        #         bn_weights_key[channel_indices_to_prune[i]] = abs_weight[channel_indices_to_prune[i]]
        #     sorted(bn_weights_key.items(), key=lambda item: item[1], reverse=True)
        #     keys = list(bn_weights_key.keys())
        #     for j in range(diff):
        #         channel_indices_to_keep.append(keys[j])
        #         channel_indices_to_prune.pop(keys[j])
        #     # np.random.shuffle(channel_indices_to_keep)
        #     # for i in range(diff):
        #     #     channel_indices_to_prune.append(channel_indices_to_keep.pop(i))
        # elif len(channel_indices_to_keep) > saved_prune_channel:
        #     print("Number of selected channels for pruning is greater than expected. Leaving too few channels.")
        #     diff = len(channel_indices_to_keep) - saved_prune_channel
        #     print("Discarding {0} pruneable channels".format(diff))
        #     bn_weights_key_2 = dict()
        #     for i in range(len(channel_indices_to_keep)):
        #         bn_weights_key_2[channel_indices_to_keep[i]] = abs_weight[channel_indices_to_keep[i]]
        #     sorted(bn_weights_key_2.items(), key=lambda item: item[1])
        #     keys = list(bn_weights_key_2.keys())
        #     for j in range(diff):
        #         channel_indices_to_prune.append(keys[j])
        #         channel_indices_to_keep.pop(keys[j])
        #     # for i in range(diff):
        #     #     channel_indices_to_keep.append(channel_indices_to_prune.pop(i))
        #
        # if len(channel_indices_to_keep) != saved_prune_channel:
        #     raise ValueError(
        #         "Number of clusters {0} is not equal with the selected "
        #         "pruneable channels {1}".format(nb_of_clusters, len(channel_indices_to_prune)))
        #####################################################################################
        return channel_indices_to_prune, channel_indices_to_keep


    def _prune_first_stage(self):
        tf_weights_value = []
        layer_name_weights_dict = dict()
        # checkpoint_path = os.path.join(self._checkpoint_dir, '')
        reader = pywrap_tensorflow.NewCheckpointReader(self._checkpoint_dir)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # prune_layer = [0,  2, 6, 9, 13, 16, 19, 22, 25, 28, 31, 34, 38, 41, 44, 47, 50, 53, 56, 59, 63, 66, 69, 72,]
        prune_darknet_layer = [0, 2, 5, 7, 10, 12,14, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35, 37, 39, 41, 44, 46, 48, 50]
        prune_head_layer = [0, 1, 2, 3, 4, 5,     8, 9, 10, 11, 12, 13,     16, 17, 18, 19, 20, 21  ]
        layer_prune_name = []
        for i in prune_darknet_layer:
            if i == 0:
                layer_prune_name.append('yolov3/darknet53_body/Conv/weights:0')
            else:
                layer_prune_name.append('yolov3/darknet53_body/Conv_' + str(i) + '/weights:0')
        for j in prune_head_layer:
            if j == 0:
                layer_prune_name.append('yolov3/yolov3_head/Conv/weights:0')
            else:
                layer_prune_name.append('yolov3/yolov3_head/Conv_' + str(j) + '/weights:0')
        for layer_name in var_to_shape_map:
            layer_name_weights_dict[layer_name + ':0'] = reader.get_tensor(layer_name)
        # print('the model dict is', layer_name_weights_dict)
        pruning_factor = self._pruning_factor
        for prune_layer in layer_prune_name:
            current_layer_name = prune_layer
            current_bn_gamma_layer = current_layer_name.replace('weights', 'BatchNorm/gamma')
            print('del current_bn_gamma_layer', current_bn_gamma_layer)
            current_gamma_layer_weight = layer_name_weights_dict[current_bn_gamma_layer]
            # print('current_gamma_layer_weight', current_gamma_layer_weight)
            filter_indices_to_prune = self._run_pruning_for_conv2d_layer(pruning_factor, current_gamma_layer_weight)
            # print('filter_indices_to_prune is ', filter_indices_to_prune)

            layer_weight = layer_name_weights_dict[prune_layer]
            W, H, N, nb_channels = layer_weight.shape
            # print("layer_weight.shape is ", layer_weight.shape)
            prune_weight = np.delete(layer_weight, filter_indices_to_prune, axis=-1)
            _, _, _, prun_channel = prune_weight.shape
            print('prun_channel is ', prun_channel)
            print('calc prune channel is ', nb_channels - len(filter_indices_to_prune))
            # print("prun weight shape is", prune_weight.shape)
            layer_name_weights_dict[prune_layer] = prune_weight

            #######prune currne layer BN params########
            bn_params = ['BatchNorm/gamma', 'BatchNorm/beta', 'BatchNorm/moving_variance',
                         'BatchNorm/moving_mean']
            bn_layer_name = []
            for i in bn_params:
                bn_params_str = current_layer_name.replace('weights', i)
                bn_layer_name.append(bn_params_str)
            for bn_layer in bn_layer_name:
                bn_param = layer_name_weights_dict[bn_layer]
                bn_filter_prune = filter_indices_to_prune
                prune_bn_param = np.delete(bn_param, bn_filter_prune, axis=0)
                layer_name_weights_dict[bn_layer] = prune_bn_param
                print('del current layer is ', bn_layer)
                print("bn param.shape is ", bn_param.shape)
                print('prune bn param shape is ', layer_name_weights_dict[bn_layer].shape)
            ###########################################

            ##### prune next input channels #############
            try:
                next_layer_number = int(prune_layer.split('/')[-2].split('_')[-1]) + 1
            except:
                next_layer_number = 1
            if 'darknet' in prune_layer:
                next_layer_name = 'yolov3/darknet53_body/Conv_' + str(next_layer_number) + '/weights:0'
            elif 'yolov3_head' in prune_layer:
                next_layer_name = 'yolov3/yolov3_head/Conv_' + str(next_layer_number) + '/weights:0'
            else:
                print(prune_layer)
                print('layer is error ')
                next_layer_name = ''
            if next_layer_number != 52:
                print("the next layer is ", next_layer_name)
                ######prune input filter weight######
                # if 'yolov3/darknet53_body/Conv/weights' not in layer_name:  ### cannot prune the first conv
                next_layer_weight = layer_name_weights_dict[next_layer_name]
                W, H, input_channels, nb_channels_2 = next_layer_weight.shape
                prun_input_next_layer_weight = np.delete(next_layer_weight, filter_indices_to_prune, axis=-2)
                layer_name_weights_dict[next_layer_name] = prun_input_next_layer_weight
                print('del next layer filter  is ', next_layer_name)
                print('orignal input shape is ', (W, H, input_channels, nb_channels_2 ))
                print('prun_input shape is ', layer_name_weights_dict[next_layer_name].shape)
                # print('calc prune channel input is ', input_channels - len(filter_indices_to_prune))
            if 'yolov3_head' in prune_layer and next_layer_number in [5, 13]:
                next_layer_name = 'yolov3/yolov3_head/Conv_' + str(next_layer_number + 2) + '/weights:0'
                print("yolo the next layer is ", next_layer_name)
                ######prune input filter weight######
                # if 'yolov3/darknet53_body/Conv/weights' not in layer_name:  ### cannot prune the first conv
                next_layer_weight = layer_name_weights_dict[next_layer_name]
                W, H, input_channels, nb_channels_2 = next_layer_weight.shape
                prun_input_next_layer_weight = np.delete(next_layer_weight, filter_indices_to_prune, axis=-2)
                layer_name_weights_dict[next_layer_name] = prun_input_next_layer_weight
                print('del next layer filter  is ', next_layer_name)
                print('orignal input shape is ', (W, H, input_channels, nb_channels_2 ))
                print('prun_input shape is ', layer_name_weights_dict[next_layer_name].shape)

        np.save('weights.npy', layer_name_weights_dict)
        return layer_name_weights_dict

    def _prune_first_stage_all_layer(self):
        tf_weights_value = []
        layer_name_weights_dict = dict()
        # checkpoint_path = os.path.join(self._checkpoint_dir, '')
        reader = pywrap_tensorflow.NewCheckpointReader(self._checkpoint_dir)
        var_to_shape_map = reader.get_variable_to_shape_map()
        # prune_layer = [0,  2, 6, 9, 13, 16, 19, 22, 25, 28, 31, 34, 38, 41, 44, 47, 50, 53, 56, 59, 63, 66, 69, 72,]
        prune_darknet_layer = [0, 2, 5, 7, 10, 12,14, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35, 37, 39, 41, 44, 46, 48, 50]
        prune_head_layer = [0, 1, 2, 3, 4, 5,     8, 9, 10, 11, 12, 13,     16, 17, 18, 19, 20, 21  ]
        layer_prune_name = []
        for i in prune_darknet_layer:
            if i == 0:
                layer_prune_name.append('yolov3/darknet53_body/Conv/weights:0')
            else:
                layer_prune_name.append('yolov3/darknet53_body/Conv_' + str(i) + '/weights:0')
        for j in prune_head_layer:
            if j == 0:
                layer_prune_name.append('yolov3/yolov3_head/Conv/weights:0')
            else:
                layer_prune_name.append('yolov3/yolov3_head/Conv_' + str(j) + '/weights:0')
        for layer_name in var_to_shape_map:
            layer_name_weights_dict[layer_name + ':0'] = reader.get_tensor(layer_name)
        # print('the model dict is', layer_name_weights_dict)
        pruning_factor = self._pruning_factor

        ##################### get th #############################################
        all_gamma_weights = []
        reshape_gamma_weights = []
        for prune_layer in layer_prune_name:
            current_layer_name = prune_layer
            current_bn_gamma_layer = current_layer_name.replace('weights', 'BatchNorm/gamma')
            all_gamma_weights.append(layer_name_weights_dict[current_bn_gamma_layer])
        sort_gamma_weights = np.sort(np.reshape(all_gamma_weights, [-1]))
        pruning_threshold = [int(np.ceil(len(sort_gamma_weights) * (1 - pruning_factor)))]

        save_channel_dict = {}
        for prune_layer in layer_prune_name:
            current_layer_name = prune_layer
            current_bn_gamma_layer = current_layer_name.replace('weights', 'BatchNorm/gamma')
            print('del current_bn_gamma_layer', current_bn_gamma_layer)
            current_gamma_layer_weight = layer_name_weights_dict[current_bn_gamma_layer]
            # print('current_gamma_layer_weight', current_gamma_layer_weight)
            # filter_indices_to_prune = self._run_pruning_for_conv2d_layer(pruning_factor, current_gamma_layer_weight)
            filter_indices_to_prune, filter_indices_to_save = self._run_pruning_for_all_layer(pruning_threshold, current_gamma_layer_weight)
            # print('filter_indices_to_prune is ', filter_indices_to_prune)
            save_channel_dict[prune_layer] = len(filter_indices_to_save)
            layer_weight = layer_name_weights_dict[prune_layer]
            W, H, N, nb_channels = layer_weight.shape
            # print("layer_weight.shape is ", layer_weight.shape)
            prune_weight = np.delete(layer_weight, filter_indices_to_prune, axis=-1)
            _, _, _, prun_channel = prune_weight.shape
            # print('prun_channel is ', prun_channel)
            print('prun channale num is', len(bn_filter_prune))
            print("kepp channel is", len(filter_indices_to_save))
            # print('calc prune channel is ', nb_channels - len(filter_indices_to_prune))
            # print("prun weight shape is", prune_weight.shape)
            layer_name_weights_dict[prune_layer] = prune_weight

            #######prune currne layer BN params########
            bn_params = ['BatchNorm/gamma', 'BatchNorm/beta', 'BatchNorm/moving_variance',
                         'BatchNorm/moving_mean']
            bn_layer_name = []
            for i in bn_params:
                bn_params_str = current_layer_name.replace('weights', i)
                bn_layer_name.append(bn_params_str)
            for bn_layer in bn_layer_name:
                bn_param = layer_name_weights_dict[bn_layer]
                bn_filter_prune = filter_indices_to_prune
                prune_bn_param = np.delete(bn_param, bn_filter_prune, axis=0)
                layer_name_weights_dict[bn_layer] = prune_bn_param
                print('del current layer is ', bn_layer)
                print("bn param.shape is ", bn_param.shape)
                print('prune bn param shape is ', layer_name_weights_dict[bn_layer].shape)
            ###########################################

            ##### prune next input channels #############
            try:
                next_layer_number = int(prune_layer.split('/')[-2].split('_')[-1]) + 1
            except:
                next_layer_number = 1
            if 'darknet' in prune_layer:
                next_layer_name = 'yolov3/darknet53_body/Conv_' + str(next_layer_number) + '/weights:0'
            elif 'yolov3_head' in prune_layer:
                next_layer_name = 'yolov3/yolov3_head/Conv_' + str(next_layer_number) + '/weights:0'
            else:
                print(prune_layer)
                print('layer is error ')
                next_layer_name = ''
            if next_layer_number != 52:
                print("the next layer is ", next_layer_name)
                ######prune input filter weight######
                # if 'yolov3/darknet53_body/Conv/weights' not in layer_name:  ### cannot prune the first conv
                next_layer_weight = layer_name_weights_dict[next_layer_name]
                W, H, input_channels, nb_channels_2 = next_layer_weight.shape
                prun_input_next_layer_weight = np.delete(next_layer_weight, filter_indices_to_prune, axis=-2)
                layer_name_weights_dict[next_layer_name] = prun_input_next_layer_weight
                print('del next layer filter  is ', next_layer_name)
                print('orignal input shape is ', (W, H, input_channels, nb_channels_2 ))
                print('prun_input shape is ', layer_name_weights_dict[next_layer_name].shape)
                # print('calc prune channel input is ', input_channels - len(filter_indices_to_prune))
            if 'yolov3_head' in prune_layer and next_layer_number in [5, 13]:
                next_layer_name = 'yolov3/yolov3_head/Conv_' + str(next_layer_number + 2) + '/weights:0'
                print("yolo the next layer is ", next_layer_name)
                ######prune input filter weight######
                # if 'yolov3/darknet53_body/Conv/weights' not in layer_name:  ### cannot prune the first conv
                next_layer_weight = layer_name_weights_dict[next_layer_name]
                W, H, input_channels, nb_channels_2 = next_layer_weight.shape
                prun_input_next_layer_weight = np.delete(next_layer_weight, filter_indices_to_prune, axis=-2)
                layer_name_weights_dict[next_layer_name] = prun_input_next_layer_weight
                print('del next layer filter  is ', next_layer_name)
                print('orignal input shape is ', (W, H, input_channels, nb_channels_2 ))
                print('prun_input shape is ', layer_name_weights_dict[next_layer_name].shape)
        np.save('weights.npy', layer_name_weights_dict)
        np.save('network_cfg,npy',l)
        return layer_name_weights_dict, prun_dict



    def _apply_fuzz(self, x):
        for i in range(len(x)):
            self.apply_fuzz_to_vector(x[i])
        return x

    def apply_fuzz_to_vector(self, x):
        # Prepare the vector element indices
        indices = np.arange(0, len(x), dtype=int)
        np.random.shuffle(indices)
        # Select the indices to be modified (always modify only N-1 values)
        nb_of_values_to_modify = np.random.randint(0, len(x) - 1)
        modify_indices = indices[:nb_of_values_to_modify]
        # Modify the selected elements of the vector
        x[modify_indices] += self._FUZZ_EPSILON

    @staticmethod
    # def _epsilon(self):
    #     return BasePruning._FUZZ_EPSILON

    def _calculate_number_of_channels_to_keep(self, keep_factor: float, nb_of_channels: int):
        # This is the number of channels we would like to keep
        # new_nb_of_channels = int(np.ceil(nb_of_channels * keep_factor))
        new_nb_of_channels = int(np.floor(nb_of_channels * keep_factor))
        if new_nb_of_channels > nb_of_channels:
            # This happens when (factor > 1)
            new_nb_of_channels = nb_of_channels
        elif new_nb_of_channels < 1:
            # This happens when (factor <= 0)
            new_nb_of_channels = 1

        # Number of channels which will be removed
        nb_channels_to_remove = nb_of_channels - new_nb_of_channels

        return new_nb_of_channels, nb_channels_to_remove

    def _save_after_pruning(self, model: models.Model):
        model.save(self._tmp_model_file_name, overwrite=True, include_optimizer=True)

    @staticmethod
    def _clean_up_after_pruning(model: models.Model):
        del model
        K.clear_session()
        tf.reset_default_graph()

    def _load_back_saved_model(self, custom_objects: dict) -> models.Model:
        model = models.load_model(self._tmp_model_file_name, custom_objects=custom_objects)
        return model
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# ########################### first prune pecentage=0.8 ###########################
sliming_model = SlimPruning(pruning_factor=0.8,
                 prune_iter_cnt=1,
                 checkpoint_dir='scale_gamma_checkpoint/best_model_Epoch_0_step_2758_mAP_0.8196_loss_3.3593_lr_1e-05')
model, prune_weights_dict = sliming_model.run_pruning()
# #################################################################################

########################### second prune pecentage=0.8 ###########################
# sliming_model = SlimPruning(pruning_factor=0.8,
#                  nb_finetune_epochs=10,
#                  prune_iter_cnt=2,
#                  checkpoint_dir="./scale_gamma_checkpoint/best_model_Epoch_0_step_3429.0_mAP_0.1628_loss_21.8372_lr_0.0001")
# model, prune_weights_dict = sliming_model.run_pruning()
#################################################################################

# ########################### third prune pecentage=0.8 ###########################
# sliming_model = SlimPruning(pruning_factor=0.8,
#                  nb_finetune_epochs=10,
#                  prune_iter_cnt=3,
#                  checkpoint_dir="./scale_gamma_checkpoint/best_model_Epoch_0_step_3429.0_mAP_0.0035_loss_47.5538_lr_0.0001")
# model, prune_weights_dict = sliming_model.run_pruning()
# #################################################################################

########################### fourth prune pecentage=0.8 ###########################
# sliming_model = SlimPruning(pruning_factor=0.8,
#                  nb_finetune_epochs=10,
#                  prune_iter_cnt=4,
#                  checkpoint_dir="./scale_gamma_checkpoint/best_model_Epoch_0_step_3429.0_mAP_0.0035_loss_47.5538_lr_0.0001")
# model, prune_weights_dict = sliming_model.run_pruning()
#################################################################################
# sliming_model._reconstruction_model(prune_weights_dict)







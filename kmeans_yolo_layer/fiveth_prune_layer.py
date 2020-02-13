
import tempfile
import os
from typing import Tuple, Callable, Union, List, Optional
from tensorflow.python.framework import graph_util
from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf
# from tensorflow.keras import backend as K
from keras import models
from sklearn import cluster, metrics
from utils.misc_utils import parse_anchors, read_class_names
from model import yolov3
from kmeans_yolo.model_kmeans import sparse_yolov3
from sliming_yolo.model_sliming import sliming_yolov3

# from ridurre import base_filter_pruning
# from ridurre import base_filter_pruning
root = '/home/pcl/tf_work/my_github/yolov3_prune'
# root = os.getcwd()
anchor_path = root + '/data/yolo_anchors.txt'  # The path of the anchor txt file.
class_name_path = root + '/data//voc.names'  # The path of the class names.
anchors = parse_anchors(anchor_path)
num_class = len(read_class_names(class_name_path))

class BasePruning:
    _FUZZ_EPSILON = 1e-5
    def __init__(self, pruning_factor: float, prune_iter_cnt: int, checkpoint_dir: str,):
        self._pruning_factor = pruning_factor
        self._tmp_model_file_name = tempfile.NamedTemporaryFile().name
        self._prune_iter_cnt = prune_iter_cnt
        self._channel_number_bins = None
        self._pruning_factors_for_channel_bins = None
        self._original_number_of_filters = -1
        self._checkpoint_dir = checkpoint_dir
        self._maximum_prune_iterations = 0

        # TODO: select a subset of layers to prune

        self._prunable_layers_regex = ".*"
        self._restore_part_first = ['yolov3/darknet53_body']
        self._restore_part_second = ['yolov3/darknet53_body','yolov3/yolov3_head']
        self._update_part = ['yolov3/yolov3_head']
        self._img_size = [608, 608]

    def run_pruning(self, prune_factor = 0.8, custom_objects_inside_model: dict = None) -> Tuple[models.Model, int]:
        # self._original_number_of_filters = self._count_number_of_filters(model)
        pruning_iteration = 0
        while True:
            if prune_factor is not None:
                self._pruning_factor = prune_factor
                # Pruning step
                print("Running filter pruning {0}".format(pruning_iteration))
                pruning_shortcut_list = list(self._prune_first_stage_layer())
                pruning_iteration += 1
                if self._maximum_prune_iterations is not None:
                    if pruning_iteration > self._maximum_prune_iterations:
                        break
            # print(pruning_shortcut_list)
            print("Pruning stopped.")
        return 0
            # return  pruning_shortcut_list

    def _prune_first_stage_layer(self):
        layer_name_weights_dict = dict()
        layer_name_mean_weights_dict = dict()
        mean_value_shortcut = dict()
        prun_shortcut_id = []
        del_layer_id_list = []
        reader = pywrap_tensorflow.NewCheckpointReader(self._checkpoint_dir)
        var_to_shape_map = reader.get_variable_to_shape_map()
        prune_darknet_layer = [2,3,  5,6,7,8,  10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,  27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,          41, 42]
        orignal_shortcut =    [0,    1,2,       3,4,5,6,7,8,9,10,                                11,12,13,14,15,16,17,                                            18]
        layer_prune_name = []
        for i in prune_darknet_layer:
            if i == 0:
                layer_prune_name.append('yolov3/darknet53_body/Conv/weights:0')
            else:
                layer_prune_name.append('yolov3/darknet53_body/Conv_' + str(i) + '/weights:0')
        for layer_name in var_to_shape_map:
            print('current layer is: ',layer_name)
            layer_name_weights_dict[layer_name + ':0'] = reader.get_tensor(layer_name)
        pruning_factor = self._pruning_factor

        for prune_layer in layer_prune_name:
            layer_weight = layer_name_weights_dict[prune_layer]
            reshape_weights = np.reshape(layer_weight, [-1])
            abs_weights = np.abs(reshape_weights)
            abs_mean_value = np.mean(abs_weights)
            layer_name_mean_weights_dict[prune_layer] = abs_mean_value
        # darknet_shortcuts = 23
        darknet_shortcuts = len(orignal_shortcut)
        short_id = 0
        short_cut = 0
        for i in range(darknet_shortcuts):
            mean_value_shortcut[i] = 0
        for layer_name, layer_value in layer_name_mean_weights_dict.items():
            mean_value_shortcut[short_id] += layer_value
            short_cut += 1
            if short_cut%2 == 0:
                short_id += 1
        print(mean_value_shortcut)
        sorted_layer_weights = sorted(mean_value_shortcut.items(), key=lambda x: x[1], reverse=False)
        print(sorted_layer_weights)

        for prune_id in range(1):
            print('prune layer ')
            c = sorted_layer_weights[prune_id][0]
            print(c)
            del_layer_id_list.append((prune_darknet_layer[2 * c]))
            del_layer_id_list.append((prune_darknet_layer[2 * c + 1]))
            print((prune_darknet_layer[2 * c]))
            print((prune_darknet_layer[2 * c + 1]))
            prun_shortcut_id.append(c)
        print("del layer id is", del_layer_id_list)
        del_layer_id_list.sort()
        print("sorted id is", del_layer_id_list)
        print(prun_shortcut_id)
        orignal_shortcut_set = set(orignal_shortcut)
        final_save = orignal_shortcut_set.difference(prun_shortcut_id)
        print(final_save)
        a = list(final_save)
        new_shortcut_list = [0, 0, 0, 0, 0]
        for i in range(22):
            if i in a:
                if i == 0:
                    new_shortcut_list[0] += 1
                if 1 <= i and i < 3:
                    new_shortcut_list[1] += 1
                if 3 <= i and i < 11:
                    new_shortcut_list[2] += 1
                if 11 <= i and i < 18:
                    new_shortcut_list[3] += 1
                if 18 <= i and i < 21:
                    new_shortcut_list[4] += 1
        result = self._reconstruction_model_prun_shortcut(new_shortcut_list, layer_name_weights_dict, del_layer_id_list)
        return final_save

    def _reconstruction_model_prun_shortcut(self, shortcut_list, weight_dict, del_layer_id_list):
        from kmeans_yolo_layer.model_kmeans import sparse_yolov3_shortcut
        print("restruction model shortcut_list ", shortcut_list)
        print("restruction model del_layer_id_list ", del_layer_id_list)
        # print("restruction model del_layer_id_list ", del_layer_id_list)
        prun_graph = tf.Graph()
        with tf.Session(graph=prun_graph) as sess:
            input_data_2 = tf.placeholder(tf.float32, [1, self._img_size[1], self._img_size[0], 3], name='input_data')
            sliming_yolo_model = sparse_yolov3_shortcut(num_class, anchors, shortcut_list)
            with tf.variable_scope('yolov3'):
                pred_feature_maps = sliming_yolo_model.forward(input_data_2, is_training=True)

            variable_names = [v.name for v in tf.trainable_variables()]

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            values = sess.run(variable_names)
            bn_params = ['BatchNorm/gamma', 'BatchNorm/beta', 'BatchNorm/moving_variance',
                         'BatchNorm/moving_mean']
            for k, v in zip(variable_names, values):
                try:
                    number = int(str(k).split('/')[2][5:])
                except:
                    number = 0
                print('current layer is', k)
                print('shape is', v.shape)
                print('current number is', number)
                name = k
                if number > del_layer_id_list[0]-1 and 'weights' in k:
                    name = 'yolov3/darknet53_body/Conv_' + str(number + 2) + '/weights:0'
                if number > del_layer_id_list[0]-1 and 'gamma' in k:
                    name = 'yolov3/darknet53_body/Conv_' + str(number + 2) + '/BatchNorm/gamma:0'
                if number > del_layer_id_list[0]-1 and 'beta' in k:
                    name = 'yolov3/darknet53_body/Conv_' + str(number + 2) + '/BatchNorm/beta:0'
                if number > del_layer_id_list[0]-1 and 'moving_variance' in k:
                    name = 'yolov3/darknet53_body/Conv_' + str(number + 2) + '/BatchNorm/moving_variance:0'
                print('current name is', name)
                current_layer_tensor = tf.get_default_graph().get_tensor_by_name(k)
                # print(weight_dict[name])
                sess.run(tf.assign(current_layer_tensor, weight_dict[name], validate_shape=True))
            print("prune network completed")
            print("completed initialized")
            saver_best = tf.train.Saver()
            saver_best.save(sess, './kmeans_checkpoint/five_kmeans_prune_restore_model_all.ckpt')
            return 0

    def run_pruning_for_conv2d_layer(self, pruning_factor: float, layer_weight_mtx):
        _, _, input_channels, nb_channels = layer_weight_mtx.shape

        # Initialize KMeans
        ########################pruning outputs#########################################
        nb_of_clusters, _ = self._calculate_number_of_channels_to_keep(pruning_factor, nb_channels)
        kmeans = cluster.KMeans(nb_of_clusters, "k-means++")

        # Fit with the flattened weight matrix
        # (height, width, input_channels, output_channels) -> (output_channels, flattened features)
        layer_weight_mtx_reshaped = layer_weight_mtx.transpose(3, 0, 1, 2).reshape(nb_channels, -1)
        # Apply some fuzz to the weights, to avoid duplicates
        layer_weight_mtx_reshaped = self._apply_fuzz(layer_weight_mtx_reshaped)
        kmeans.fit(layer_weight_mtx_reshaped)

        # If a cluster has only a single member, then that should not be pruned
        # so that point will always be the closest to the cluster center
        closest_point_to_cluster_center_indices = metrics.pairwise_distances_argmin(kmeans.cluster_centers_,
                                                                                    layer_weight_mtx_reshaped)
        #print('closest_point_to_cluster_center_indices is ', closest_point_to_cluster_center_indices)
        # Compute filter indices which can be pruned
        channel_indices = set(np.arange(len(layer_weight_mtx_reshaped)))
        channel_indices_to_keep = set(closest_point_to_cluster_center_indices)
        channel_indices_to_prune = list(channel_indices.difference(channel_indices_to_keep))
        channel_indices_to_keep = list(channel_indices_to_keep)

        if len(channel_indices_to_keep) > nb_of_clusters:
            print("Number of selected channels for pruning is less than expected")
            diff = len(channel_indices_to_keep) - nb_of_clusters
            print("Randomly adding {0} channels for pruning".format(diff))
            np.random.shuffle(channel_indices_to_keep)
            for i in range(diff):
                channel_indices_to_prune.append(channel_indices_to_keep.pop(i))
        elif len(channel_indices_to_keep) < nb_of_clusters:
            print("Number of selected channels for pruning is greater than expected. Leaving too few channels.")
            diff = nb_of_clusters - len(channel_indices_to_keep)
            print("Discarding {0} pruneable channels".format(diff))
            for i in range(diff):
                channel_indices_to_keep.append(channel_indices_to_prune.pop(i))

        if len(channel_indices_to_keep) != nb_of_clusters:
            raise ValueError(
                "Number of clusters {0} is not equal with the selected "
                "pruneable channels {1}".format(nb_of_clusters, len(channel_indices_to_prune)))
        #####################################################################################
        return channel_indices_to_prune

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
    def _epsilon(self):
        return BasePruning._FUZZ_EPSILON

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


    def _load_back_saved_model(self, custom_objects: dict) -> models.Model:
        model = models.load_model(self._tmp_model_file_name, custom_objects=custom_objects)
        return model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
check_root = '/home/pcl/tf_work/my_github/yolov3_prune/kmeans_yolo_layer/kmeans_checkpoint/'
checkpoint_dir = check_root + 'fourth_kmeans_prune_restore_model_all.ckpt'
# checkpoint_dir = check_root + 'best_model_Epoch_7_step_22071_mAP_0.8267_loss_3.5250_lr_0.0001'

### first   shortcut list [1, 2, 8, 7, 4]
### second  shortcut list [1, 2, 8, 7, 3]
### third   shortcut list [1, 2, 8, 7, 2]
### fiveth   shortcut list [1, 2, 8, 7, 1]
parse_model = BasePruning(pruning_factor=0.8, prune_iter_cnt=1,checkpoint_dir=checkpoint_dir)
prune_weights_dict = parse_model.run_pruning()


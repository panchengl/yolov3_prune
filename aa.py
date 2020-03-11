# a =[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22]
# b = 0
# new_shortcut_list = [0, 0, 0, 0, 0]
# new_shortcut_list_2 = [0, 0, 0, 0, 0]
#
# if 0 in a:
#     new_shortcut_list[0] += 1
# if 1 in a:
#     new_shortcut_list[1] += 1
# if 2 in a:
#     new_shortcut_list[1] += 1
# if 3 in a:
#     new_shortcut_list[2] += 1
# if 4 in a:
#     new_shortcut_list[2] += 1
# if 5 in a:
#     new_shortcut_list[2] += 1
# if 6 in a:
#     new_shortcut_list[2] += 1
# if 7 in a:
#     new_shortcut_list[2] += 1
# if 8 in a:
#     new_shortcut_list[2] += 1
# if 9 in a:
#     new_shortcut_list[2] += 1
# if 10 in a:
#     new_shortcut_list[2] += 1
# if 11 in a:
#     new_shortcut_list[3] += 1
# if 12 in a:
#     new_shortcut_list[3] += 1
# if 13 in a:
#     new_shortcut_list[3] += 1
# if 14 in a:
#     new_shortcut_list[3] += 1
# if 15 in a:
#     new_shortcut_list[3] += 1
# if 16 in a:
#     new_shortcut_list[3] += 1
# if 17 in a:
#     new_shortcut_list[3] += 1
# if 18 in a:
#     new_shortcut_list[3] += 1
# if 19 in a:
#     new_shortcut_list[4] += 1
# if 20 in a:
#     new_shortcut_list[4] += 1
# if 21 in a:
#     new_shortcut_list[4] += 1
# if 22 in a:
#     new_shortcut_list[4] += 1
# print(new_shortcut_list)
#
# for i in range(23):
#     if i in a:
#         if i == 0:
#             new_shortcut_list_2[0] += 1
#         if 1 <= i and i < 3:
#             new_shortcut_list_2[1] += 1
#         if 3 <= i and i < 11:
#             new_shortcut_list_2[2] += 1
#         if 11 <= i and i < 19:
#             new_shortcut_list_2[3] += 1
#         if 19 <= i and i < 23:
#             new_shortcut_list_2[4]+= 1
# print(new_shortcut_list_2)
# layer_prune_name = []
# prune_darknet_layer = [2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30,
#                        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51]
# for i in prune_darknet_layer:
#     if i == 0:
#         layer_prune_name.append('yolov3/darknet53_body/Conv/weights:0')
#     else:
#         layer_prune_name.append('yolov3/darknet53_body/Conv_' + str(i) + '/weights:0')
# a = [11, 19, 21, 12, 20]
# b = []
# print(layer_prune_name)
#
# for prune_id in range(5):
#     print('prune layer ')
#     c = a[prune_id]
#     b.append((prune_darknet_layer[2*c] ))
#     b.append((prune_darknet_layer[2*c+1] ))
#     print((prune_darknet_layer[2*c] ))
#     print((prune_darknet_layer[2*c + 1]) )
#
# # first = dict()
# # second = dict()
# # third = dict()
# # fourth = dict()
# # last = dict()
# first = []
# second = []
# third = []
# fourth = []
# last = []
# import copy
# # for i in b:
# #     layer_prune_name.remove('yolov3/darknet53_body/Conv_' + str(i) + '/weights:0')
# first = copy.deepcopy(layer_prune_name)
# first.remove('yolov3/darknet53_body/Conv_' + str(27) + '/weights:0')
# first.remove('yolov3/darknet53_body/Conv_' + str(28) + '/weights:0')
# first.remove('yolov3/darknet53_body/Conv_' + str(44) + '/weights:0')
# first.remove('yolov3/darknet53_body/Conv_' + str(45) + '/weights:0')
# first.remove('yolov3/darknet53_body/Conv_' + str(48) + '/weights:0')
# first.remove('yolov3/darknet53_body/Conv_' + str(49) + '/weights:0')
# first.remove('yolov3/darknet53_body/Conv_' + str(29) + '/weights:0')
# first.remove('yolov3/darknet53_body/Conv_' + str(30) + '/weights:0')
# first.remove('yolov3/darknet53_body/Conv_' + str(46) + '/weights:0')
# first.remove('yolov3/darknet53_body/Conv_' + str(47) + '/weights:0')
# # layer_prune_name.remove('yolov3/darknet53_body/Conv_' + str(27) + '/weights:0')
# for i, j in enumerate(layer_prune_name):
#     print(str(j).split('/')[2][5:])
#     # first[i] = j
#     # if int(str(j).split('/')[2][5:]) >= b[0]:
#     #     first[i] = 'yolov3/darknet53_body/Conv_' + str(int(str(j).split('/')[2][5:]) -1) + '/weights:0'
#     # if int(str(j).split('/')[2][5:]) >= b[1]:
#     #     first[i] = 'yolov3/darknet53_body/Conv_' + str(int(str(j).split('/')[2][5:]) - 1) + '/weights:0'
# # second = copy.deepcopy(first)
# # for i, j in enumerate(first):
# #     print(str(j).split('/')[2][5:])
# #     first[i] = j
# #     if int(str(j).split('/')[2][5:]) >= b[0]:
# #         first[i] = 'yolov3/darknet53_body/Conv_' + str(int(str(j).split('/')[2][5:]) -1) + '/weights:0'
#
# print(layer_prune_name)
# print(first)
#
# c = set(layer_prune_name)
# d = c.difference(first)
# print(d)


a =[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22]

b = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 22]

import numpy as np
print(np.multiply(np.array(a), np.array(b)))
# print(a*b)
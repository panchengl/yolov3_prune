import xml.etree.ElementTree as ET
from os import getcwd
import os
sets=[('dianli', 'test')]

classes = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe", "SuLiaoBu", "FengZheng", "Niao", "NiaoWo", "ShanHuo", "YanWu", "JianGeBang", "JueYuanZi", "FangZhenChui"]
# classes = [ "ShanHuo", "YanWu"]
# classes = [ "SuLiaoBu", 'yiwu']
# classes = ["DiaoChe", "TaDiao", "TuiTuJi", "BengChe", "WaJueJi", "ChanChe"]
# img_dir = "/home/pcl/data/VOC2007/JPEGImages"
# anno_dir = "/home/pcl/data/VOC2007/Annotations"
# /home/pcl/data/voc/VOCdevkit

img_dir = "/home/pcl/data/VOC2007/JPEGImagesTest"
anno_dir = "/home/pcl/data/VOC2007/AnnotationsTest"

def convert_annotation(year, image_id, list_file):
    # in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    # anno_id =
    print(image_id)
    in_file = open(anno_dir+ '/%s.xml'% image_id.split('.')[0])
    tree=ET.parse(in_file)
    root = tree.getroot()
    for size in root.iter('size'):
        width = int(size.find('width').text)
        height = int(size.find('height').text)

    list_file.write(" " + str(width) + " " + str(height))
    for obj in root.iter('object'):
        try:
            difficult = obj.find('difficult').text
        except:
            difficult = 0
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
            ######### if normal , del this code ###########3333
        if cls in classes:
            xmlbox = obj.find('bndbox')
            b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
                 int(xmlbox.find('ymax').text))
            list_file.write(" " + str(0) + " " + " ".join([str(a) for a in b]))
        ################################################
        # try:
        #     cls_id = classes.index(cls)
        # except:
        #     continue
        # xmlbox = obj.find('bndbox')
        # b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        # list_file.write(" " + str(cls_id) + " " + " ".join([str(a) for a in b]))

wd = getcwd()

for year, image_set in sets:
    # image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    image_ids = os.listdir(img_dir)
    print("[INFO] the current img nums is %d" %len(image_ids))
    num = 0
    list_file = open('./data/my_data/%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        # print(image_id)
        #num = num + 1
        if(num %1000 == 0):
            print("current deal img_num is %d"%num)
        if image_id.split('.')[-1] != 'jpg':
            continue
        list_file.write(str(num) + ' ' + img_dir + '/%s' %(image_id))
        convert_annotation(year, image_id, list_file)
        num = num + 1
        list_file.write('\n')
    list_file.close()


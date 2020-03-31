20200311 updates:

    add mnn inference code, when i use mmn converted tools, some ops donn't support , so i need change some code for success converted(pb->mnn). finally, i test tf inference results and mnn
    
     inference results, there is a difference of four digits after the decimal point.
     
     --reference project
     
                1. https://github.com/alibaba/MNN
                2. https://github.com/wlguan/MNN-yolov3   (android code)

    add quant code from google paper, but this code just fusion bn and conv layer, and the code doesn't train, i have no time to fix it , and this fusion code was plagiarized, ha ha ha, if i have another time , i wiil fixed it

20200214 updates:

    add structure prune--layer prune algorithm, compared with the channel pruning algorithm, the efficiency is not high. ---Reference Paper DeepCompression SongHan

20200117 updates:

    add gussian-yolov3, map 0.8378->0.8459 in voc dataset

20200110 updates:

    add soft-prune, but cannot use
    
    add spp1, spp3 backbone


# yolov3_prune

  In this project, you can end-to-end pruning models.It should be noted that my project is based on others projects,so,if you want prune yolov3 model, you can use the following steps:
  
  first stage:
    git clone https://github.com/wizyoung/YOLOv3_TensorFlow
    
  second stage:
    Train the model according to his steps， just like create your dataset, and use: python train.py, then, you will obtained a yolov3 model.
    
  third stage:
    Then, use my project, put my project file overwrite his project files, in pruning_kneans_yolov3.py tihs file, modify your model path and your saveed path, at last use: python pruning_kneans_yolov3.py, you will obtain a prune model ,default prune ratio is 0.8, if you want other rations, you can edit params.

  last stage:
  
    Fine tune mode, python finetune.py
    
   results:
   
   thsi is my prune models and map in voc:
   https://github.com/panchengl/yolov3_prune/orignal_results.png
   
   my trained voc data results:
   
   orignal model map is 0.835 input size is 416x416, inference time is 23ms in 1080Ti, model size is 240M。
   
   after first prune，map is 0.82, input size is 416x416, inference time is 20ms in 1080Ti, model size is 210M
   
   
I referred to many code：

  https://github.com/wizyoung/YOLOv3_TensorFlow
  
   and so on,I have forgotten because it has been around for a long time.so I will not listed one by one.
   
   
   
   
   
   
    

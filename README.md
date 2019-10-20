# yolov3_prune
  In this project, you can end-to-end pruning models.It should be noted that my project is based on others projects,so,if you want prune yolov3 model, you can use the following steps:
  
  first stage:
    git clone https://github.com/wizyoung/YOLOv3_TensorFlow
    
  second stage:
    Train the model according to his steps， just like create your dataset, and use: python train.py, then, you will obtained a yolov3 model.
    
  third stage:
    Then, use my project, put my project file overwrite his project files, in pruning_kneans_yolov3.py tihs file, modify your model path and your saveed path, at last use: python pruning_kneans_yolov3.py, you will obtain a prune model ,default prune ratio is 0.8, 
if you want other rations, you can edit params.

  last stage:
    Fine tune mode, python finetune.py
    
   results:
   thsi is my prune models and map in voc:
   https://github.com/panchengl/yolov3_prune/orignal_results.png
   
   my trained voc data results:
   orignal model map is 0.57, input size is 416x416, inference time is 23ms in 1080Ti, model size is 240M。
   after first prune, 
   map is 0.54, input size is 416x416, inference time is 20ms in 1080Ti, model size is 210M
   
   
I referred to many code：
  https://github.com/wizyoung/YOLOv3_TensorFlow
   and so on,I have forgotten because it has been around for a long time.so I will not listed one by one.
   
   
   
   
   
   
    

from __future__ import print_function

import sys
sys.path.append('../convertor-code')
sys.path.append('/home/pcl/tf_work/')

import numpy as np
import MNN
import cv2
from create_mnn_pb import postprocess_doctor_yang
# from v3.utils.tools import img_preprocess2
# from v3.utils.tools import draw_bbox
# from v3.pb import postprocess

INPUTSIZE=608
CLASSES = []
img_dir = '/home/pcl/tf_work/YOLOv3_TensorFlow/dianli_608/mnn/convertor-code/v3/val00003.jpg'
def inference():
    """ inference mobilenet_v1 using a specific picture """
    interpreter = MNN.Interpreter("/home/pcl/tf_work/YOLOv3_TensorFlow/dianli_608/mnn/convertor-code/v3/doctor_yang_7.mnn")
    # interpreter = MNN.Interpreter("doctor_yang_7_quan.mnn")
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    # image = cv2.imread('test/val00027.jpg')
    # orishape = image.shape
    # originimg=image.copy()

    # image=img_preprocess2(image, None, (INPUTSIZE, INPUTSIZE), False)[np.newaxis, ...]

    img_ori = cv2.imread(img_dir)
    orishape = img_ori.shape
    originimg=img_ori.copy()
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple([INPUTSIZE, INPUTSIZE]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    image = img[np.newaxis, :] / 255.

    #cv2 read shape is NHWC, Tensor's need is NCHW,transpose it
    tmp_input = MNN.Tensor((1,INPUTSIZE, INPUTSIZE,3), MNN.Halide_Type_Float, image, MNN.Tensor_DimensionType_Tensorflow)
    #construct tensor from np.ndarray
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    output_tensor = interpreter.getSessionOutput(session)
    output_data=np.array(output_tensor.getData())
    print('output data is', output_data)
    output_data=output_data.reshape((-1,11))
    print('output data is', output_data)
    outbox = np.array(postprocess_doctor_yang(output_data, INPUTSIZE, orishape[:2]))
    print('result box is', outbox)
    # originimg = draw_bbox(originimg, outbox, CLASSES)
    # cv2.imwrite('result.jpg', originimg)

if __name__ == "__main__":
    with open('ours.names', 'r') as f:
        CLASSES = f.readlines()
    inference()
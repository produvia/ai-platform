# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 07:54:51 2019

@author: LEGION
"""

""" This module for converting darknet models to keras models has been extended for Spatial pyramid pooling as well as any available dataset 
by simply downloading the appropriate weights, specifying the dataset and setting the SPP flag. Spatial Pyramid pooling technique makes a number 
of modifications to the middle of YOLO layers as implemented here.
"""


from download_file import download_if_not_exists
import sys
#sys.path.append(r"C:\Users\LEGION\tf1")
import struct
import mlflow
import mlflow.keras
import os
import numpy as np
from keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, UpSampling2D,MaxPool2D
from keras.layers.merge import add, concatenate
from keras.models import Model


def _conv_block(inp, convs, skip=True):
  x = inp
  count = 0
  len_convs = len(convs)
  for conv in convs:
    if count == (len_convs - 2) and skip:
      skip_connection = x
    count += 1
    if conv['stride'] > 1: x = ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
    x = Conv2D(conv['filter'],
           conv['kernel'],
           strides=conv['stride'],
           padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
           name='conv_' + str(conv['layer_idx']),
           use_bias=False if conv['bnorm'] else True)(x)
    if conv['bnorm']: x = BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
    if conv['leaky']: x = LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)
  return add([skip_connection, x]) if skip else x


#SPP block uses three pooling layers of sizes [5, 9, 13] with strides one and all outputs together with the input are concatenated to be fed
  #to the FC block
def _SPP_block(inp, kernels, strides):
    pools = [MaxPool2D(pool_size = pool_size, strides = stride, padding = 'same')(inp) \
             for pool_size, stride in zip(kernels, strides)]
    pools = [inp] + pools
    return concatenate(pools)


#Downsampling block is common to all YOLO-v3 models and are unaffected by the SPP or fully connected blocks or the number of labes
def downSample(input_image):
      # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])
  # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
            {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
            {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])
  # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
            {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])
  # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
            {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])
  # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
              {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
    skip_36 = x
  # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])
  # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
              {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
    skip_61 = x
  # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
            {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])
  # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
              {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
        
    return skip_36, skip_61, x

#The upsampling block is affected by the number of clases to be predicted as the number of filters for the YOLO layers = 3(num_classes +5)
def upSampling(x, skip_36, skip_61, layer_idx, num_classes=80):
    out_filters = 3*(num_classes+5)
    yolo_83 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': layer_idx},
            {'filter':  out_filters, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': layer_idx+1}], skip=False)

    x = _conv_block(x, [{'filter':  256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': layer_idx+4}],\
                    skip = False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_61])
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': layer_idx+7},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': layer_idx+8},
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': layer_idx+9},
            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': layer_idx+10},
            {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': layer_idx+11}], skip=False)
  # Layer 92 => 94
    yolo_95 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True, \
                               'layer_idx': layer_idx+12},
                {'filter': out_filters, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': layer_idx+13}],\
                          skip=False)
  # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': layer_idx+16}],\
                    skip=False)
    x = UpSampling2D(2)(x)
    x = concatenate([x, skip_36])
  # Layer 99 => 106
    yolo_107 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True, \
                                'layer_idx':layer_idx+19},
                 {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': layer_idx+20},
                 {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': layer_idx+21},
                 {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': layer_idx+22},
                 {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': layer_idx+23},
                 {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': layer_idx+24},
                 {'filter': out_filters, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': layer_idx+25}],\
                           skip=False)
    
    
    return yolo_83, yolo_95, yolo_107

#The midblock is where the spatial pyramid pooling as well as the FC block with change for the YOLOv3-SPP model are reflected
def midBlock(x, SPP= True):
    if SPP:
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
            {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77}], skip = False)
        x = _SPP_block(x, [5,9,13], [1,1,1])

        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79},
                {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 80}], skip = False)
        return x
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)
    return x

def make_yolo(num_classes=80, SPP = True):
    input_image = Input(shape = (None, None, 3))
    skip_36, skip_61, x = downSample(input_image)
    x = midBlock(x, SPP)
    start_id = 81 if SPP else 80
    yolo_82, yolo_94, yolo_106 = upSampling(x, skip_36, skip_61, start_id, num_classes)
    
    return Model(input_image, [yolo_82, yolo_94, yolo_106])
        
        
 #weightReader asjusted to accomodate SPP layers as well as various dataset sizes   
class WeightReader: 
    
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,  = struct.unpack('i', w_f.read(4))
            minor,  = struct.unpack('i', w_f.read(4))
            revision, = struct.unpack('i', w_f.read(4))
            transpose = (major > 1000) or (minor > 1000)
            if (major*10 + minor) >= 2 and not transpose:
                w_f.read(8)
            else:
                w_f.read(4)
            
            binary = w_f.read()
            self.offset = 0
            self.all_weights = np.frombuffer(binary, dtype='float32')

    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]

    def load_weights(self, model, SPP= True):
        num_Layers = 107 if SPP else 106
        yolo_layers = [82, 94, 106] if SPP else [81, 93, 105]
        conv_count = 0
        for i in range(num_Layers):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))
                if i not in yolo_layers:  
                    norm_layer = model.get_layer('bnorm_' + str(i))
                    size = np.prod(norm_layer.get_weights()[0].shape)
                    #print("norm size: {}".format(size))
                    beta  = self.read_bytes(size) # bias
                    gamma = self.read_bytes(size) # scale
                    mean  = self.read_bytes(size) # mean
                    var   = self.read_bytes(size) # variance
                    print("Norm loaded")
                    norm_layer.set_weights([gamma, beta, mean, var])
                if len(conv_layer.get_weights()) > 1:
                    bias   = self.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
                    print("bias loaded")
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    print("weight loaded")
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel, bias])
                else:
                    kernel = self.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
                    print("weight loaded")
                    kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
                    kernel = kernel.transpose([2,3,1,0])
                    conv_layer.set_weights([kernel])
                conv_count +=1        
            except ValueError:
                print("no convolution #" + str(i))
        print("Loaded {} convolution layers".format(conv_count))

    def reset(self):
        self.offset = 0
        
def dataset_process(data_path):
    with open(data_path, "r") as f:
        num_classes = int(f.readline().split("=")[-1])
        classes = f.read().strip().split("\n")
    return num_classes, classes

        

class YOLO:
    
    def __init__(self, dataset = "openimages", SPP = False):
        if SPP and dataset != "coco":
            print("SPP weights only exist for coco dataset. Reverting to ordinary yolov3...")
        self.SPP = SPP if dataset =="coco" else False
        appendfunc = lambda spp, dataset: "-spp" if spp else "" if dataset == "coco" else "-{}".format(dataset)
        self.data = 'data/{}.data'.format(dataset)
        self.url = 'https://pjreddie.com/media/files/yolov3{}.weights'.format(appendfunc(self.SPP, dataset))
        self.darknet_model_path = 'models/yolov3{}.weights'.format(appendfunc(self.SPP, dataset))
        self.kerasPath = 'models/yolov3{}.h5'.format(appendfunc(self.SPP, dataset))
        self.num_classes, self.classes = dataset_process(self.data)
        self.model = make_yolo(self.num_classes, self.SPP)
        #print(self.model.summary())
        
    def load_darknet_weights(self, darknet_model_path = None, kerasPath = None):
        
        if kerasPath == None: kerasPath = self.kerasPath
        if darknet_model_path == None : darknet_model_path = self.darknet_model_path
        if not os.path.exists(kerasPath):
            weight_reader = WeightReader(darknet_model_path)
            weight_reader.load_weights(self.model, self.SPP)
            self.model.save(kerasPath)
            print("Model saved at: {}".format(kerasPath))
        else:
            print("Model already exists")
            self.load_model(kerasPath)
            
    def download_weights(self, darknet_model_path, url):
        if darknet_model_path ==None: darknet_model_path = self.darknet_model_path
        url = url if url else self.url
        download_if_not_exists(darknet_model_path, url)
        
        
    def load_model(self, model_path):
        self.model.load_weights(model_path)
        print("Model loaded from: {}".format(model_path))
        
if __name__ =="__main__":
    url = sys.argv[5] if len(sys.argv)>5 and sys.argv[5]!="None" else None
    darknet_model_path = sys.argv[1] if len(sys.argv)>1 and sys.argv[1]!="None" else None
    dataset = sys.argv[2] if len(sys.argv)>2 else 'openimages'
    SPP = int(sys.argv[3]) if len(sys.argv)>3 else 1
    print("SPP use: ", SPP)
    kerasPath = sys.argv[4] if len(sys.argv)>4 and sys.argv[4] !="None" else None
    yolo = YOLO(dataset, SPP)
    yolo.download_weights(darknet_model_path, url)
    yolo.load_darknet_weights(darknet_model_path, kerasPath)
    
    
    with mlflow.start_run():
        mlflow.log_param("darknet_model_path", yolo.darknet_model_path)
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("SPP", yolo.SPP)
        mlflow.log_param("darknet_url", yolo.url)
        mlflow.keras.log_model(yolo.model, yolo.kerasPath)
        

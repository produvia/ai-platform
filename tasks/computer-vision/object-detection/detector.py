# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 14:06:38 2019

@author: LEGION
"""
import sys
from yolo_image import process_image
from yolo_video import process_video
if __name__ =="__main__":
    #print(sys.argv)
    mode = sys.argv[1] if len(sys.argv) > 1 else 'image'
    keras_model_path = sys.argv[2] if len(sys.argv) > 2 else 'models/yolov3-openimages.h5'
    size = int(sys.argv[3]) if len(sys.argv) >3 else 416
    assert size%32 ==0 and size/32 >= 10 , "Size Error: Enter a multiple of 32 greater or equal to 320. size {} invalid!".format(size)
    dataset = sys.argv[4] if len(sys.argv) > 4 else 'openimages'
    if mode =="video":
        print("video")
        item = sys.argv[5] if len(sys.argv) > 5 else 'data/Working.mp4'
        process_video(keras_model_path, size, dataset, item)
    elif mode =="image":
        print("image")
        item = sys.argv[5] if len(sys.argv) > 5 else 'data/dog.jpg'
        process_image(keras_model_path, size, dataset, item)
    else:
        raise Exception("{} mode is not valid!. Use\"image\" or \"video\".")
        
        
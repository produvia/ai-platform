# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 20:14:50 2019

@author: LEGION
"""

import os
#import sys
#sys.path.append(r"C:\Users\LEGION\tf1")
import numpy as np
import cv2
from yolo_image import cvDrawBoxes, decode_netout, correct_yolo_boxes, do_nms, get_boxes
from yolov3_weights_to_keras import dataset_process
import time
import mlflow
import mlflow.keras
from keras.models import load_model



def process_video(keras_model_path='models/yolov3-spp.h5', size = 320, dataset = "coco", video_path = "data/Working.mp4"):
    dataset = "data/{}.data".format(dataset)
    cv2.namedWindow("Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Demo", 720, 1280)
    input_w, input_h = size, size
    if not os.path.exists("outputs"):os.mkdir("outputs")
    model = load_model(keras_model_path)
    cap = cv2.VideoCapture(video_path)
    image_h, image_w = int(cap.get(4)), int(cap.get(3))
    print(image_h, image_w)
    output_video_name = "outputs/object_" + os.path.basename(video_path)
    #cap.set(3, 1280)
    #cap.set(4, 720)
    
    out = cv2.VideoWriter(
        output_video_name, cv2.VideoWriter_fourcc(*"MJPG"), 10.0,(image_w, image_h))
    print("YOLO starting")
    done = False
    frame_no = 0
    with mlflow.start_run(nested = True):
        mlflow.log_param("keras_model_path", keras_model_path)
        mlflow.log_param("video_name", os.path.basename(video_path))
        while not done:
            try:
                prev_time = time.time()
                ret, frame = cap.read()
                frame_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_h, image_w, _ = frame_img.shape
                frame = cv2.resize(frame,(input_w,input_h),interpolation=cv2.INTER_LINEAR)
                frame= frame.astype(np.float32) / 255.0 
                y_hat = model.predict(np.expand_dims(frame, 0))
                anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
                class_threshold = 0.6
                boxes = list()
                for i in range(len(y_hat)):
        #            # decode the output of the network
                    boxes += decode_netout(y_hat[i][0], anchors[i], class_threshold, input_h, input_w)
        #        # correct the sizes of the bounding boxes for the shape of the image
                correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
        #        # suppress non-maximal boxes
                do_nms(boxes, 0.5)
        #        # define the labels
                _,labels = dataset_process(dataset)
        #        
                v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)
                for i in range(len(v_boxes)):
                    print(v_labels[i], v_scores[i])
                    tag_name = "{}_{}_{}".format(frame_no, i, v_labels[i])
                    mlflow.set_tag(tag_name, v_scores[i])
                frame_img = cvDrawBoxes(frame_img, v_boxes, v_labels, v_scores)
            
 
                frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
                print(1/(time.time()-prev_time))
                cv2.imshow('Demo', frame_img)
                out.write(frame_img)
                cv2.waitKey(3)
                frame_no +=1
            except:
                done = True
                print("Done!")
    print("Video created after object detection task and saved as:", output_video_name)
    mlflow.log_artifact(output_video_name, "output_video_name")
    
    cv2.destroyAllWindows()
    cap.release()
    out.release()
if __name__ =="__main__":
    process_video()       
            
            
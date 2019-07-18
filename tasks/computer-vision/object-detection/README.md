# Object Detection
 
Detect object in given image/video using any dataset: coco, openimages, voc. Default is set to coco dataset with the categories:
- aeroplane
- apple
- backpack
- banana
- baseball bat
- baseball glove
- bear
- bed
- bench
- bicycle
- bird
- boat
- book
- bottle
- bowl
- broccoli
- bus
- cake
- car
- carrot
- cat
- cell phone
- chair
- clock
- cow
- cup
- diningtable
- dog
- donut
- elephant
- fire hydrant
- fork
- frisbee
- giraffe
- hair drier
- handbag
- horse
- hot dog
- keyboard
- kite
- knife
- laptop
- microwave
- motorbike
- mouse
- orange
- oven
- parking meter
- person
- pizza
- pottedplant
- refrigerator
- remote
- sandwich
- scissors
- sheep
- sink
- skateboard
- skis
- snowboard
- sofa
- spoon
- sports ball
- stop sign
- suitcase
- surfboard
- teddy bear
- tennis racket
- tie
- toaster
- toilet
- toothbrush
- traffic light
- train
- truck
- tvmonitor
- umbrella
- vase
- wine glass
- zebra

The data folder contains the labels for the openimages, voc and coco labels. Custom datasets of choice can also be used simply by following the format provided in the data files
Also included is the option of using Spatial Pyramid pooling as in yolov3-spp simply by setting the parameter to 1

### Running Example
- Navigate to tasks/computer-vision/object-detection/ directory within the repository.
- Place the image/video in same data directory for object detection and issue any of the following commands:

For image data run:

    `mlflow run . -P file_name=data/horses.jpg`
- Or specifiy absolute path of the image as following:

    `mlflow run . -P file_name='/home/user/Downloads/sheeps.jpeg'`

To make use of any dataset e.g openimages data, simply add an extra parameter as follows:

    `mlflow run . -P file_name=data/dog.jpg -P dataset=openimages`

To make use of Spatial Pyramid Pooling architecture, simply add an extra parameter as follows:

    `mlflow run . -P file_name=data/dog.jpg -P dataset=openimages -P SPP=1`

To control the size(320, 352, 416 608 etc..Default is 416)...Note that size must be a multiple of 32 and be >=320:

    `mlflow run . -P file_name=data/dog.jpg -P dataset=openimages -P SPP=1 -P size=608`

To supply the url for pre-trained weights of any darknet model, provided the corresponding data file is provided appropriately in the data folder:

    `mlflow run . -P file_name=data/dog.jpg -P darknet_url='https://pjreddie.com/media/files/yolov3.weights'`

To process videos, simply add an extra parameter as follows(include mode=video and use the video path as file-name, all other configurations stay the same):

    `mlflow run . -P file_name=data/video.mp4 -P dataset=openimages -P mode=video`

### TODO: model serving as image is not featured yet into mlflow, need to work on an alternate approach.
### TODO: Include option for training on custom dataset of any size with categories not found in any of the provided datasets
### TODO: Alternatives to speed up post-processing, the present pipeline presents too much of a lag

### NOTE: The default url for loading voc weights is unavailable, however if you have a url with pre-trained darknet yolo weights on voc. You can simply add to the parameters
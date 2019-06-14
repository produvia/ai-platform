# Object Detection
 
Detect object in given image in 80 different categories.
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

### Running Example
- Navigate to tasks/computer-vision/object-detection/ directory within the repository.
- Place the image in same direcotry for object detection and issue following command:

    `mlflow run . -P photo_name=zebra.jpg`
- Or specifiy absolute path of the image as following:

    `mlflow run . -P photo_name='/home/user/Downloads/sheeps.jpeg'`

### TODO: model serving as image is not featured yet into mlflow, need to work on an alternate approach.
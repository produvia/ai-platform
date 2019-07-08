# Super Resolution using convolutional neural networks.

Increase image resolution x2 or x4 while maintaining or even improving image quality.

## Usage

### Train:
List of parameters:\n
  data_path: Directory of training images. Make sure to keep all images in a folder named 'train' before running script.
  size: Size of final image after super resolution. e.g. 1024 means the model will learn to turn a smaller image to (1024,1024)
  factor: Upscale factor (2 or 4). If size  = 1024 and upscale factor = 4, model will learn to turn image from (256,256) to (1024,1024)

mlflow run . -e train -P data_path='/training_data/' -P size=200 -P factor=2 -P bs=16 -P epochs=50 -P device='cuda:0'

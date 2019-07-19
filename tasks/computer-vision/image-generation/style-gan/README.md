# StyleGAN Pytorch Implementation
This is a Pytorch implementation of StyleGAN, following the official implementation as much as possible.

## Prerequisites
### Image generation using official implementation's checkpoints

**If downloading fails for Google Drive, manual download is required:**

* [ffhq-1024x1024](https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ)
* [bedrooms-256x256](https://drive.google.com/open?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF)
* [cats-256x256](https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ)

And place them in ./pretrained directory.

### Training prerequisites on CelebA dataset
Download the [celeba](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg) dataset. Unzip the .zip file into ./data/celeba directory.

## Image Generation

Run the command:
```bash
mlflow -e generate . -P dataset=cats
The default random seed is 77. To generate different images with different image grid (note that the number of images you can generate is limited by your GPU).
```bash
mlflow -e generate -P datasets=cats -P random_seed=777 -P nrow=2 -P ncol=5
```
This will generate 15 images at once.

## Training on CelebA dataset
Run the command:
```bash
mlflow -e train . -P data_root='./data/celeba' -P resolution=128 -P n_gpus=1
```
This will kick off the training for 128x128 resolution on CelebA dataset. During training, the model checkpoints are stored under ./checkpoints, and the fake images are generated for checking under ./checks/fake\_imgs. Note that this is a progressive process starting from 8x8, so you will see 8x8 images in the begining and 128x128 images in the end of the training process. 


## TODO
Due to the time limitation, I didn't implement the following:
1. Add truncation trick
2. Add and experiment other loss functions
3. Add tensorboard support
4. Add moving average of generator's weight

**All of the above are not essential to get visually good results, but are implemented in the official tensorflow implemenation.**

Multi-GPU support is added but not experimented due to hardware limitation.

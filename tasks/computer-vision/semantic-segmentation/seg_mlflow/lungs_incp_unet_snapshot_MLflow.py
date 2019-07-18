
# coding: utf-8

# # Image Segmentation using an Inception based U-Net model with snapshots and cosine annealed learning rate.
# 
# This notebook is an example notebook of using image data and its corresponding masks to produce masks for the test images whose mask are needed.
# 
# Segmentation of an image is to classify each pixel of an image into 2 or more classes. For problems where only one object of interest is needed to be determined in an image containing the object and background, we need to classify every pixel into 2 classes as is the case with the dataset/ model used in this notebook.  

import os
import sys
# ## Import packages
# 
# Import different packages for handling images and building the keras model.


# Packages for processing/ loading image files
from glob import glob 
import cv2
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

# Packages for the model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, BatchNormalization
from keras.layers import LeakyReLU, Concatenate, concatenate, UpSampling2D, Add, Input,Dense
from keras.models import Model,load_model
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard

# Import the snapshot builder callback file locally which is used for dynamic LR.
from snapshot import SnapshotCallbackBuilder

# Package for model deployment 
import click
import mlflow
import mlflow.keras


# ## Function to load and process Image
# 
# Load image using opencv function imread(path, 1 for read as colour image).
# 
# Cv2 loads the image with B channel first. We converte this to R channel first.
# 
# Gaussian Blur is applied to image to have smoother transitions in image. 
# 
# Image is scaled to have values between 0 and 1.
# 
# For processing masks it is made sure that shapes are as required and the image is thresholded to converte arbitary values to 0 or 255. 


# Preprocess image for of top view training/ detection
def preprocess_img(img_file):
    im = cv2.imread(img_file, 1)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.GaussianBlur(im, (5,5), 0)
    im = im/255
    return im

# Preprocess masks top view
def preprocess_mask(img_file):
    im = cv2.imread(img_file, 0)
    im = im.reshape(im.shape[0], im.shape[1], 1)
    _, im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)
    im = cv2.GaussianBlur(im,(5,5), 0)
    im = im.reshape(im.shape[0], im.shape[1], 1)
    im = im/255
    return im


# ## Inception Block
# 
# Instead of a single convolution block an inception block is used to extract the features from the image.
# 
# In an inception block input is divided into multiple instances and passed to different combinations of convolution layer and then the output from all the combinations are concatenated and passed further.
# 
# An inception block extracts richer feature compared to a convolution layer since in an inception block same input feature is passed through different sized kernel of convolutions, which enables the model to process the features from different point of views. 
# 
# Different combinations of layers can be experimented with. Also different inception blocks can be combined in a same model and used as done in the actual inception model.
# 
# Link to the paper for Inception model is [here](https://arxiv.org/pdf/1602.07261.pdf).
def incp_v3(x, nb_filters):

    b_1 = Conv2D(nb_filters, (1,1), padding='same', kernel_initializer="glorot_normal")(x)
    b_1 = BatchNormalization(axis=3)(b_1)
    b_1 = LeakyReLU()(b_1)
    b_1 = Conv2D(nb_filters, (3,3), padding='same', kernel_initializer="glorot_normal")(b_1)
    b_1 = BatchNormalization(axis=3)(b_1)
    b_1 = LeakyReLU()(b_1)
    b_1 = Conv2D(nb_filters, (3,3), padding='same', kernel_initializer="glorot_normal")(b_1)
    b_1 = BatchNormalization(axis=3)(b_1)
    b_1 = LeakyReLU()(b_1)
	
    b_2 = Conv2D(nb_filters, (1,1), padding='same', kernel_initializer="glorot_normal")(x)
    b_2 = BatchNormalization(axis=3)(b_2)
    b_2 = LeakyReLU()(b_2)
    b_2 = Conv2D(nb_filters, (3,3), padding='same', kernel_initializer="glorot_normal")(b_2)
    b_2 = BatchNormalization(axis=3)(b_2)
    b_2 = LeakyReLU()(b_2)
	
    b_3 = Conv2D(nb_filters, (1,1), padding='same', kernel_initializer="glorot_normal")(x)
    b_3 = BatchNormalization(axis=3)(b_3)
    b_3 = LeakyReLU()(b_3)

    b_4 = MaxPooling2D((2,2), strides=(1,1), padding='same')(x)
    b_4 = Conv2D(nb_filters, (1,1), padding='same', kernel_initializer="glorot_normal")(b_4)
    b_4 = BatchNormalization(axis=3)(b_4)
    b_4 = LeakyReLU()(b_4)
    
    x = concatenate([b_1, b_2, b_3, b_4], axis=3)
    return x

# image_augmenting function
def image_augmentation(imgs, masks, batch_size): 
	data_gen_args = dict(featurewise_center=False,
						 featurewise_std_normalization=False,
						 zoom_range = [0.9, 1.1],
						 width_shift_range=0.1,
						 height_shift_range=0.1,
						 horizontal_flip=True,
						 vertical_flip = True ) 

	image_datagen = ImageDataGenerator(**data_gen_args)
	mask_datagen = ImageDataGenerator(**data_gen_args)
	seed = 1

	image_datagen.fit(imgs, augment=True, seed=seed)
	mask_datagen.fit(masks, augment=True, seed=seed)

	image_generator = image_datagen.flow(
		imgs,
		batch_size=batch_size,
		shuffle=True,
		seed=seed)

	mask_generator = mask_datagen.flow(
		masks,
		batch_size=batch_size,
		shuffle=True,
		seed=seed)

	train_generator = zip(image_generator, mask_generator)

	return train_generator


def get_model(inp):
	x = incp_v3(inp, 8) 

	b_0 = x

	x = MaxPooling2D((2, 2), strides=2)(x) 
	b_1 = (x)

	x = incp_v3(x, 16) 

	x = MaxPooling2D((2, 2), strides=2)(x) 
	b_2 = (x) 

	x = incp_v3(x, 32) 

	x = MaxPooling2D((2, 2), strides=2)(x) 
	b_3 =(x) 

	x = incp_v3(x, 64) 

	encoded = MaxPooling2D((2, 2))(x) 

	x = incp_v3(encoded, 64)

	x = UpSampling2D((2, 2))(x) 
	x = Concatenate(axis=3)([x, b_3])

	x = incp_v3(x, 32)

	x = UpSampling2D((2, 2))(x) 
	x = Concatenate(axis=3)([x, b_2])

	x = incp_v3(x, 16)

	x = UpSampling2D((2, 2))(x) 
	x = Concatenate(axis=3)([x, b_1])

	x = incp_v3(x, 8)

	x = UpSampling2D((2, 2))(x) 
	x = Concatenate(axis=3)([x, b_0])

	decoded = Conv2D(1, (3,3), activation='sigmoid', padding='same', kernel_initializer='glorot_normal')(x)
	
	return decoded
	
	
# Callback class for logging metrics into the MLflow pipeline.
class mlfow_log_callback(tf.keras.callbacks.Callback):

	def on_epoch_end(self, epoch, logs=None):
		mlflow.log_metric('Train Loss',logs['loss'], epoch)
		mlflow.log_metric('Train Accuracy',logs['acc'], epoch)
		mlflow.log_metric('Validation Loss',logs['val_loss'], epoch)
		mlflow.log_metric('Valdiation Accuracy',logs['val_acc'], epoch)
		print(epoch, logs)
		

# click provides an easy interface for getting cmd line arguments.
@click.command()
@click.option("--image_path", default='../finding-lungs-in-ct-data/2d_images/', help="Path to images folder")
@click.option("--annotation_path", default='../finding-lungs-in-ct-data/2d_masks/', help="Path to annotations folder")
@click.option("--weights_path", default='../weights/', help="Path to base model weights file")
@click.option("--log_dir", default='../logs/', help="Path to store log files")
@click.option("--initial_lr", default=1e-3, type=float, help="Initial learning rate")
@click.option("--batch_size", default=4, type=int, help="Batch size for training")
@click.option("--seed", default=3, type=int, help="numpy random seed")
def train(image_path, annotation_path, weights_path, log_dir, initial_lr, batch_size, seed):

	# ## Dataset
	# 
	# The dataset for this notebook is taken from -> https://www.kaggle.com/kmader/finding-lungs-in-ct-data/home .
	# 
	# The dataset contains 267 images of ct scans of lungs with masks representing the lung portion in each of the image.
	# 
	# The images in the dataset are in tif format.
	# Load the training files
	#
	X = []
	file_train = glob(image_path+"*.tif")
	for i, file in enumerate(file_train):
			im = preprocess_img(file)
			X.append(im)
	X = np.array(X)

	y = []
	file_mask = glob(annotation_path+"*.tif")
	for i in file_mask:
			im = preprocess_mask(i)
			y.append(im)  
	y = np.array(y)

	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)

	x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=seed)

	# ## Auto Encoder with U-Net Architecture
	# 
	# In an auto encoder model for an image, features are extracted and reduced using a convolution network (or another feature extractor like using inception blocks). The feature size is reduced by max pooling features from each kernel in the convolution. Since, over here, our aim is to get the mask of the input image, we reconstruct the latent features (features found after the last max pooling layer), into an image of same size (channel size may differ). This is done by upsampling (copying the same value in each of the kernel position) and convolution of these features. 
	# 
	# To increase the accuracy of the network an architecture known as U-Net was used. On the decoder side where reconstruction of image is done, along with latent features, corresponding features from encoder is concatenated in each layer. These allows the decoder network to directly look at the features of corresponding layers in the encoder, to enchance feature reproduction. The paper on [U-Net architecture](https://arxiv.org/abs/1505.04597) gives detail information on the same.  
	#
	# Initalize the Inception based UNET model

	inp = Input(X[0].shape)
	decoded = get_model(inp)

	train_generator = image_augmentation(x_train, y_train, batch_size)
	val_generator = image_augmentation(x_val, y_val, batch_size)

	tb = TensorBoard(log_dir=log_dir+"lungs_incp_unet_snapshot_mlflow", histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True,
			 embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

	# Model parameters
	M = 3
	nb_epoch = T = 60
	snapshot = SnapshotCallbackBuilder(T, M, initial_lr)
	timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
	model_prefix = 'lungs_incp_unet_snapshot_mlflow{}'.format(timestr)

	# Train model
	callbacks = [tb] + snapshot.get_callbacks(model_prefix = model_prefix) + [mlfow_log_callback()]


	with mlflow.start_run() as run:

		model = Model(inp, decoded)
		model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

		hist = model.fit_generator(train_generator, steps_per_epoch=int(len(x_train)/batch_size), epochs=nb_epoch, callbacks=callbacks, verbose=0,
							validation_data=val_generator, validation_steps=int(len(x_val)/batch_size))


		## may give location you want to save the model to.
		model.save(weights_path+'lungs_incp_unet_snapshot_mlflow')
		
		pred = [model.predict(x.reshape((1,x.shape[0],x.shape[1],x.shape[2])))[0] for x in x_test]

		temp = []
		for im in y_test:
			im *=255
			_, im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)
			temp.append(im)
		y_test_t = np.array(temp)

		temp = []
		for im in pred:
			im *=255
			_, im = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)
			temp.append(im)
		pred_t = np.array(temp)

		# ## IOU
		# 
		# Intersection Over Union is used to evaluate the models performance.
		# 
		# The predicted image is compared with the corresponding mask of the test image.


		component1 = np.float32(y_test_t)
		component2 = pred_t
		overlap = np.logical_and(component1, component2) # Logical AND
		union = np.logical_or(component1, component2) # Logical OR

		IOU = overlap.sum()/float(union.sum())
		print(IOU)
		
		# Log Parameters and Test score for a run of the model.
		mlflow.log_param("batch_size", str(batch_size))
		mlflow.log_param("seed", str(seed))
		mlflow.log_metric('Test IOU',IOU)
		mlflow.log_param("initial_lr", str(initial_lr))
		# Log the model.
		mlflow.keras.log_model(model, "lungs_incp_unet_snapshot_mlflow_log")
		
		
if __name__ == '__main__':
    train()
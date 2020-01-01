"""Trains and evaluates a simple CNN for object classification on the CIFAR10 dataset"""
import argparse

import keras
import mlflow.keras
import numpy as np
import pandas  as pd
from keras import layers, models, optimizers, preprocessing, regularizers
from keras.datasets import cifar10
from matplotlib import pyplot, pyplot as plt
from numpy import random
from sklearn import metrics


def build_model(num_classes, input_shape):
	"""Create a CNN-based classification model.

	Inspired by https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/

	:param num_classes: number of output classes
	:param input_shape: shape of the input images
	:return: keras model
	"""

	input = layers.Input(input_shape)
	x = input
	dropout = 0.4
	for kernel, filters, pool, dropout_cnn in (
			(3, 32, 2, dropout),
			(3, 32, 1, dropout),
			(3, 64, 2, dropout),
			(3, 64, 1, dropout),
			(3, 128, 2, dropout),
			(3, 128, 1, dropout)):
		initializer = keras.initializers.glorot_uniform()
		regularizer = regularizers.l2(1e-4)
		if kernel and filters:
			x = layers.Conv2D(filters, (kernel, kernel), activation='relu', padding='same',
							  kernel_initializer=initializer, kernel_regularizer=regularizer)(x)
			x = layers.BatchNormalization()(x)
			if pool > 1:
				x = layers.MaxPooling2D(pool_size=(pool, pool))(x)
			x = layers.Dropout(dropout_cnn)(x)
	x = layers.Flatten()(x)
	predictions = layers.Dense(num_classes, activation='softmax')(x)

	# this is the model we will train
	model = models.Model(inputs=input, outputs=predictions)

	# compile the model (should be done *after* setting layers to non-trainable)
	model.compile(
			optimizer=optimizers.Adam(decay=0.0001),
			loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def prepare_data():
	"""Load and preprocess the CIFAR10 dataset.

	:return: (x_train, y_train), (x_test, y_test), label_names
	"""
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	# Normalize data to [0, 1]
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255

	num_classes = y_train.max() + 1

	# Binarize the labels
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	labels = "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"  # from https://github.com/EN10/CIFAR

	return (x_train, y_train), (x_test, y_test), labels


def train(model, batch_size, epochs, augment):
	"""Train a keras model with given parameters."""

	if augment:
		datagen = preprocessing.image.ImageDataGenerator(
				rotation_range=15,
				width_shift_range=0.1,
				height_shift_range=0.1,
				horizontal_flip=True,
		)
		datagen.fit(x_train)
		model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
							steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs,
							verbose=1, validation_data=(x_test, y_test))
	else:
		model.fit(x_train, y_train, epochs=epochs, validation_split=0.05, batch_size=batch_size)


def analyze(y_test, y_pred, labels):
	"""Analyze the predictions results."""

	cm = metrics.confusion_matrix(np.argmax(y_test, axis=1), y_pred)

	labels = 'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'
	df_cm = pd.DataFrame(cm, labels, labels)
	try:
		import seaborn as sn
		plt.figure(figsize=(10, 7))
		sn.set(font_scale=1.4)
		sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, fmt='g')
		plt.title('Label confusion matrix')
		plt.show()
	except ImportError:
		print(df_cm)


def show_images(x, correct, predicted):
	"""Show an array of images with their labels."""

	pyplot.figure(figsize=(10, 7))
	k = 0
	cols = 4
	rows = x.shape[0] // cols
	for i in range(rows):
		for j in range(cols):
			pyplot.subplot2grid((rows, cols), (i, j))
			pyplot.imshow(x[k])
			pyplot.grid(None)
			pyplot.axis('off')
			if correct[k] == predicted[k]:
				pyplot.title(correct[k])
			else:
				pyplot.title(f'{predicted[k]} ({correct[k]})', color='r')
			k += 1
	pyplot.show()


# Command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=20,
                    help='number of epochs to train. 150 should get you to ~85% accuracy')
parser.add_argument('--augment', type=bool, default=True,
                    help='Do image augmentation for training?')
args = parser.parse_args()

# Automatically log metrics and parameters to MLflow.
mlflow.keras.autolog()

(x_train, y_train), (x_test, y_test), labels = prepare_data()
input_shape = x_train.shape[1:]
num_classes = y_train.shape[1]

model = build_model(num_classes, input_shape)
model.summary()
train(model, args.batch_size, args.epochs, args.augment)  # 150 epochs should achieve ~85% accuracy in the validation set

y_pred = np.argmax(model.predict(x_test, verbose=2), axis=1)

analyze(y_test, y_pred, labels)

# Take a few random images and plot them, including the labels
selected_images = random.randint(x_test.shape[0], size=16)
show_images(x_test[selected_images],
			[labels[np.argmax(y_test[idx])] for idx in selected_images],
			[labels[y_pred[idx]] for idx in selected_images])

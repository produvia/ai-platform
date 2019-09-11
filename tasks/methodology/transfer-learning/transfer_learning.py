import warnings
import sys
import os

from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
from keras.applications import VGG16

from keras import models
from keras import layers
from keras.callbacks import EarlyStopping

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Read command line inputs
    samples_per_class = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    retrain_cnn_layers = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    use_early_stop = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    max_epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    with mlflow.start_run():

        # log mlflow parameters
        mlflow.log_param("retrain_cnn_layers", retrain_cnn_layers)  
        mlflow.log_param("samples_per_class", samples_per_class)
        mlflow.log_param("use_early_stop", use_early_stop)  
        mlflow.log_param("max_epochs", max_epochs)

        # load CIFAR10 dataset
        (train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

        # select subset of training samples (per class)
        number_of_classes = 10
        train_data_subset = np.array([])
        train_labels_subset = np.array([])
        for i in range(number_of_classes):
            mask = np.where(train_labels == i)[0]
            if i == 0:
                train_data_subset = train_data[mask[:samples_per_class]]
                train_labels_subset = train_labels[mask[:samples_per_class]]
            else:
                train_data_subset = np.concatenate((train_data_subset,train_data[mask[:samples_per_class]]))
                train_labels_subset = np.concatenate((train_labels_subset,train_labels[mask[:samples_per_class]]))


        # transform data (normalisation, categorisation of labels and train-test-validate split)
        x_train = train_data_subset.astype('float32')/255
        test_size = int(test_data.shape[0]*0.75)
        x_test = test_data[:test_size].astype('float32')/255
        x_val = test_data[test_size:].astype('float32')/255

        y_train = to_categorical(train_labels_subset)
        y_test = to_categorical(test_labels[:test_size])
        y_val = to_categorical(test_labels[test_size:])
        
        # build transfer learning model
        # base model: VGG16 (use only convolutional layers)
        # VGG16 is trained with the ImageNet dataset (thousands of output classes in object classification)
        conv_base = VGG16(weights='imagenet',
                        include_top=False,
                        input_shape=(32,32,3))
        # decide how many cnn layers will be retrained (fine tuning of the network)
        # last 'retrain_cnn_layers' will be retrained
        do_training = True
        current_cnn_layer = 0
        for layer in reversed(conv_base.layers):
            if 'conv' in layer.name:
                if current_cnn_layer >= retrain_cnn_layers:
                    do_training = False
                current_cnn_layer += 1
            layer.trainable = do_training
            print('layer {} is now trainable {}'.format(layer.name,layer.trainable))

        # add new dense layers specific to CIFAR10 classifier (10 output classes)
        pre_trained_model = models.Sequential()
        pre_trained_model.add(conv_base)
        pre_trained_model.add(layers.Flatten())
        pre_trained_model.add(layers.Dense(256, activation='relu'))
        pre_trained_model.add(layers.Dense(number_of_classes, activation='softmax'))

        # compile model for training
        pre_trained_model.compile(optimizer='rmsprop',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

        
        # train the model with training data (subset of total)
        # use the validation datasets to monitor training
        if use_early_stop > 0:
            # use early stop to avoid overfitting
            callbacks = [EarlyStopping(monitor='val_loss', patience=2)]
            history = pre_trained_model.fit(x_train,
                                            y_train,
                                            epochs=max_epochs,
                                            callbacks=callbacks,
                                            batch_size=32,
                                            validation_data=(x_val,y_val))
        else:
            # train with fixed number of epochs
            history = pre_trained_model.fit(x_train,
                                            y_train,
                                            epochs=max_epochs,
                                            batch_size=32,
                                            validation_data=(x_val,y_val))
        

        # evaluate model performance with the unseen test dataset
        test_loss, test_acc = pre_trained_model.evaluate(x_test,y_test)

        # output results
        print('Test accuracy {}, Test loss {}'.format(test_acc,test_loss))

        # saving model weights
        output_folder = './model_weights'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        pre_trained_model.save_weights(output_folder + '/model.h5')

        # log mlflow metrics
        mlflow.log_metric("training_set", train_data_subset.shape[0])
        mlflow.log_metric("validation_set", x_test.shape[0])
        mlflow.log_metric("test_set", x_test.shape[0])
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)


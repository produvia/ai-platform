# Imports
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.models import Model
from keras.layers import Dense, CuDNNLSTM, Input
import datetime
import os
import shutil
from numpy.random import seed
from tensorflow import set_random_seed
import matplotlib.pyplot as plt
import argparse
import mlflow.sklearn
from mlflow import log_metric
import mlflow.keras
import warnings
import sys

# Initialize
# Keras and Tensorflow Random Numbers
seed(1)
set_random_seed(2)
now = datetime.datetime.utcnow()

# Clear
# Previous Tensorflow Session
tf.keras.backend.clear_session()
tf.logging.set_verbosity(tf.logging.ERROR)

# Define Variables
arr = sys.argv
lookback_window = 32
steps_per_epoch = 364 * 10
epochs = 30
val_steps = 364 * 2
LSTMunits = 100
predict_steps = 364 * 4
predict_shift = 0
city = sys.argv[1]
print ('Running Weathernet for '+city)
# Working directories
model_filepath = 'Weather_Net_'+city+'.hf5'
image_dir = 'images/'

# Load Dataset - Daily climate data Hamburg 1951-01-01 - 2018-12-31
# 0 Mess_Datum (date of measurement) -
# 1 Wind_max (wind max)
# 2 Wind_mittel (wind average)
# 3 Niederschlagshoehe (precipitation height)
# 4 Sonnenstunden (hours of sunshine)
# 5 Schneehoehe (snow height)
# 6 Bedeckung_Stunden (cloud cover in hours)
# 7 Luftdruck (air pressure)
# 8 Temp_mittel (temp average)
# 9 Temp_max (temp max)
# 10 Temp_Min (temp min)
# 11 Temp_Boden (temp ground)
# 12 Relative_Feuchte (relative humidity)
dataset = np.loadtxt(city+'.csv', dtype='float32', delimiter=';')
print (dataset.shape[0])
# Chose which variable to predict - in this case 9 - Temp_max
variable_to_forecast = 9
single_val_ds = dataset[:, variable_to_forecast].T
single_val_ds = np.reshape(single_val_ds, newshape=(dataset.shape[0], 1))

# Transform features to range (-1, 1) with sklearn MinMaxScaler
scaler2 = MinMaxScaler(feature_range=(-1, 1))
scaler2.fit(single_val_ds)
scaled_sv_ds = scaler2.transform(single_val_ds)
dataset = scaled_sv_ds

# Fit generators for keras NN - sliding window
# Training Generator
def datagen_train():
    while 1:
        for i in range(lookback_window, lookback_window + steps_per_epoch + 1):
            train_x = dataset[i - lookback_window:i, 0]
            train_x_reshape = np.reshape(train_x, newshape=(1, lookback_window, 1))
            train_y = dataset[i + 1, 0]
            train_y_reshape = np.reshape(train_y, newshape=(1, 1))
            yield ({'seq_input': train_x_reshape}, {'output_1': train_y_reshape})

# Validation Generator
def datagen_val():
    while 1:
        for i in range(lookback_window + steps_per_epoch, lookback_window + steps_per_epoch + val_steps + 1):
            train_x = dataset[i - lookback_window:i, 0]
            train_x_reshape = np.reshape(train_x, newshape=(1, lookback_window, 1))
            train_y = dataset[i + 1, 0]
            train_y_reshape = np.reshape(train_y, newshape=(1, 1))
            yield ({'seq_input': train_x_reshape}, {'output_1': train_y_reshape})

# Prediction Generator
def datagen_predict():
    while 1:
        for i in range(lookback_window + steps_per_epoch + val_steps, lookback_window + steps_per_epoch + val_steps + predict_steps + 1):
            # print i
            train_x = dataset[i - lookback_window:i, 0]
            train_x_reshape = np.reshape(train_x, newshape=(1, lookback_window, 1))
            train_y = dataset[i + 1, 0]
            train_y_reshape = np.reshape(train_y, newshape=(1, 1))
            yield ({'seq_input': train_x_reshape}, {'output_1': train_y_reshape})

# Build and compile LSTM sequence prediction model
def build_and_compile_model():
    seq_input = Input(shape=(lookback_window, 1), name='seq_input', batch_shape=(1, lookback_window, 1))
    x = CuDNNLSTM(LSTMunits, kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform', return_sequences=True)(seq_input)
    x = CuDNNLSTM(LSTMunits, kernel_initializer='glorot_uniform', recurrent_initializer='glorot_uniform', return_sequences=False)(x)
    output_1 = Dense(1, activation='linear', name='output_1')(x)

    weathernet = Model(inputs=seq_input, outputs=output_1)
    weathernet.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss='mse')
    weathernet.summary()
    return weathernet

# Predict
def predict_weather(trained_weathernet):
    yhat = trained_weathernet.predict_generator(datagen_predict(), steps=predict_steps)
    ground_truth = np.reshape(dataset[lookback_window + steps_per_epoch + val_steps+1: lookback_window + steps_per_epoch + val_steps + predict_steps + 1, 0], newshape=(predict_steps, 1))
    plot_series(yhat, ground_truth)

# Plot prediction vs. ground-truth and save image for MLflow
def plot_series(yhat, ground_truth):
    (fig, ax) = plt.subplots()
    ax.plot(scaler2.inverse_transform(ground_truth), color='green', label="Truth")
    ax.plot(scaler2.inverse_transform(yhat), color='blue', label="Prediction")
    plt.title('Daily Temperature [max] '+city)
    plt.legend(loc='best')
    plt.savefig(image_dir+str(now)+'_'+city+'_Daily_Temp_Predicted.png')
    plt.show()

# Plot metrics and save figure for MLflow
def plot_metrics(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    (fig, ax) = plt.subplots()
    ax.plot(loss, color='green', label="Loss")
    ax.plot(val_loss, color='blue', label="Validation Loss")
    plt.title('Training Loss/Validation Loss '+city)
    plt.legend(loc='best')
    plt.savefig(image_dir +str(now)+'_'+city+ '_Loss_Diag.png')
    plt.show()

# Get loss from keras history object
def loss(hist):
    loss = hist.history['loss']
    loss_val = loss[len(loss) - 1]
    return loss_val

# Get validation loss from keras history object
def val_loss(hist):
    val_loss = hist.history['val_loss']
    validation_loss = val_loss[len(val_loss)-1]
    return validation_loss

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    #Build model
    weathernet = build_and_compile_model()
    #Train model
    results = weathernet.fit_generator(datagen_train(), steps_per_epoch=steps_per_epoch, workers=10, max_queue_size=100, epochs=epochs, verbose=2, validation_steps=val_steps, validation_data=datagen_val())
    #Save trained model
    weathernet.save(filepath=model_filepath)
    # Run prediction on trained model
    predict_weather(weathernet)
    # Plot the metrics of the trained model
    plot_metrics(results)
    # Log metrics, parameters, artifacts and log the model
    with mlflow.start_run():
        run_uuid = mlflow.active_run().info.run_uuid
        print("MLflow Run ID: %s" % run_uuid)
        mlflow.keras.log_model(weathernet, "models")
        mlflow.log_artifact(image_dir + str(now) + '_' + city + '_Loss_Diag.png', "images")
        mlflow.log_artifact(image_dir + str(now) + '_' + city + '_Daily_Temp_Predicted.png', "images")
        mlflow.log_metric('Loss', loss(results))
        mlflow.log_metric('Validation Loss', val_loss(results))
        mlflow.log_param('City_Name', city)
        mlflow.log_param('Training_Epochs', epochs)
        mlflow.log_param('Steps_per_epoch', steps_per_epoch)
        mlflow.log_param('Validations_steps', val_steps)
        mlflow.log_param('Prediction_steps', predict_steps)
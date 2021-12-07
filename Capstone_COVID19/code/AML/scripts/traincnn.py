# Numeric libs
import numpy as np

# Plotting libs
import matplotlib.pyplot as plt
import seaborn as sns

# Metrics libs
from sklearn.metrics import confusion_matrix

# TensorFlow libs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import l1, l2, l1_l2

# Azure ML Libraries
import azureml.core
from azureml.core import Dataset, Run
from azureml.core import Workspace, Experiment, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.core.model import Model
from azureml.core import Model

# Argument Parsing
import argparse

# Custom callback metric - logs epoch metrics back to Azure ML
class LogToAzure(tf.keras.callbacks.Callback):
    def __init__(self, run):
        super(LogToAzure, self).__init__()
        self.run = run

    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.run.log(k, v)

# Mount ADLS and pull Azure ML Workspace context
def adls_mount():
    ws = Run.get_context().experiment.workspace
    dataset_name = 'adls'
    adlsdataset = Dataset.get_by_name(workspace=ws, name=dataset_name)
    mount_context = adlsdataset.mount()
    mount_context.start()
    return ws, mount_context 

# Parse inbound parameters from SubmitTrainingJob
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--filters',
            type=int,
            help='# Convolutional Filters'
        )

    parser.add_argument(
            '--densenodes',
            type=int,
            help='# Dense Nodes'
        )

    parser.add_argument(
            '--modelname',
            type=str,
            help='Name of the model'
        )
    args = parser.parse_args()
    Run.get_context().log('Filters', args.filters)
    Run.get_context().log('Dense Layers', args.densenodes)
    return args

# Load NumPy arrays from ADLS
def load_arrays():
    path=mount_context.mount_point+"/TrainTestSetsBalanced"
    X_train = np.load(path+'/X_train.npy')
    X_test = np.load(path+'/X_test.npy')
    y_train = np.load(path+'/y_train.npy')
    y_test = np.load(path+'/y_test.npy')
    return X_train, y_train, X_test, y_test

# Define the sequential model architecture
def conv_arch():

    logcallback = LogToAzure(Run.get_context())
    model = Sequential()
    model.add(Conv2D(args.filters, (3, 3), input_shape=X_train.shape[1:], kernel_regularizer=l1_l2(l1=.01,l2=.01)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())  
    model.add(Dense(args.densenodes, kernel_regularizer=l1(.01)))
    model.add(Dense(3))
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                optimizer='adam',
                metrics=['accuracy'])
    return logcallback, model

# Train the model and save it into Azure ML Models
def train(model, args, X_train, y_train, epochs, split, shuffle):
    history = model.fit(X_train, 
                        y_train,
                        epochs=epochs,
                        validation_split=split,
                        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', 
                        patience=3),logcallback], shuffle = shuffle)
    model.save('outputs/{}.h5'.format(args.modelname))
    return model, history

# Predict Test set data and save results/confusion matrix
def predictions(X_test, y_test):

    y_pred = []

    for i in X_test:
        image = np.expand_dims(i, axis=0)
        pred = np.argmax(model.predict(image))
        y_pred.append(pred)

    cf = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12,10))
    heatmap = sns.heatmap(cf, annot=True,cmap='Blues', fmt='g')
    heatmap.set_xticklabels(['covid','healthy','pneumonia'])
    heatmap.set_yticklabels(['covid','healthy','pneumonia'])
    Run.get_context().log_image(name='Confusion Matrix', plot=plt)
    return

# Run script
ws, mount_context = adls_mount()
args = parse_arguments()
X_train, y_train, X_test, y_test = load_arrays()
logcallback, model = conv_arch()
model, history = train(model, args, X_train, y_train, epochs = 30, split = 0.2, shuffle = True)
predictions(X_test, y_test)

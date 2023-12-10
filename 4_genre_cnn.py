# Author: Sophie Zhao

''' 
https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8?gi=ac21c958a822
https://www.analyticsvidhya.com/blog/2021/06/music-genres-classification-using-deep-learning-techniques/
Reference : https://www.youtube.com/c/ValerioVelardoTheSoundofAI
'''

import json
from pickletools import optimize
from matplotlib import units
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

# dataset location
INPUT_DATA_PATH = "./data_with_labels.json"

# 25% of the dataset will be used as test set
TEST_RATIO = 0.25

# 20% of the train set will be used as validation set
VALIDATION_RATIO = 0.2

# Drop out ratio
DROP_OUT = 0.3

# Load data
def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    # Convert lists into numpy arrays
    inputs = np.array(data["mfcc"])          
    targets = np.array(data["labels"])       
    
    return inputs, targets

def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Training Evaluation")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("")

    plt.show()

def process_datasets(test_size, validation_size):
    # Load data from the json file generated from MFCC
    inputs, targets = load_data(INPUT_DATA_PATH)
    
    # Train/Test split
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, 
                                                                              targets, 
                                                                              test_size=test_size)
    
    # Train/Validation split
    inputs_train, inputs_validation, targets_train, targets_validation = train_test_split(inputs_train, 
                                                                                          targets_train, 
                                                                                          test_size=validation_size)

    # Tensorflow expect 3d array for CNN samples. The current dataset is 2d, so 1d need to be added
    inputs_train = inputs_train[..., np.newaxis]
    inputs_validation = inputs_validation[..., np.newaxis]
    inputs_test = inputs_test[..., np.newaxis]
    
    return inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test

def build_model(input_shape):
    # Create model
    cnn_model = keras.Sequential()
    
    # 1st Convolutional layer + pooling
        # Filters(kenel): 32
        # Grid size of the kenel: 3x3    
        
    cnn_model.add(keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape ))
    cnn_model.add(keras.layers.MaxPool2D((3,3), strides = (2,2), padding = 'same'))
    cnn_model.add(keras.layers.BatchNormalization())
    
    # 2nd Convolutional layer + pooling
    cnn_model.add(keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = input_shape ))
    cnn_model.add(keras.layers.MaxPool2D((3,3), strides = (2,2), padding = 'same'))
    cnn_model.add(keras.layers.BatchNormalization())
    
    # 3rd Convolutional layer + pooling
    cnn_model.add(keras.layers.Conv2D(32, (2,2), activation = 'relu', input_shape = input_shape ))
    cnn_model.add(keras.layers.MaxPool2D((2,2), strides = (2,2), padding = 'same'))
    cnn_model.add(keras.layers.BatchNormalization())
    
    # Flatten and add a dense layer
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(64, activation = 'relu'))
    cnn_model.add(keras.layers.Dropout(0.3))
    
    # Output layer
    cnn_model.add(keras.layers.Dense(10, activation = 'softmax'))
    
    return cnn_model
    
def predict(X,y):
    new_model = tf.keras.models.load_model('saved_model/my_model')
    
    # model.predict expect a 4 dementional array
    # x is currently 3 dementional like (130,13,1)
    # We need to change it to (1,130,13,1)
    
    X = X[np.newaxis, ...]
    prediction = new_model.predict(X)
    
    # The result of the prediction will be a two dimentional array like [[0.1, 0,2, ...]]
    
    
    # To get the top 1: extract index with max value    
    predicted_index = np.argmax(prediction, axis=1) # will get the label number e.g. : [4]
    print("The expect index is {}, the prediction index is {}".format(y, predicted_index));

def run():
    # Load data
    # inputs, targets = load_data(DATASET_PATH)
    
    # Generate sets for train, validation and tests 
        #   Use validation set to optimize the model    
        #   Test sets will never be exposed to the model before end testing  
    inputs_train, inputs_validation, inputs_test, targets_train, targets_validation, targets_test = process_datasets(0.25, VALIDATION_RATIO)
    
    
    # Build CNN model
    input_shape = (inputs_train.shape[1], inputs_train.shape[2], inputs_train.shape[3])
    cnn_model = build_model(input_shape)
    
    # Compile the model 
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    cnn_model.compile(optimizer=optimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics=["accuracy"]
                  )

    cnn_model.summary()
    
    # Train the CNN model
    history = cnn_model.fit(inputs_train, 
              targets_train, 
              validation_data=(inputs_validation, targets_validation),
              epochs=80,
              batch_size=32)
    
    # Evaluate the CNN model against the test sets
    test_error, test_accuracy = cnn_model.evaluate(inputs_test, targets_test, verbose = 2)
    print("Test accuracy against the CNN model is: {}".format(test_accuracy))
    
    # plot accuracy/error for training and validation
    plot_history(history)
    
    # Save the entire model as a SavedModel.
    cnn_model.save('saved_model/my_model')
    
if __name__ == "__main__":
    run()
    



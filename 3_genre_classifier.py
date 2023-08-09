# Author: Sophie Zhao

''' 
Reference : https://www.youtube.com/c/ValerioVelardoTheSoundofAI
https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8?gi=ac21c958a822
https://www.analyticsvidhya.com/blog/2021/06/music-genres-classification-using-deep-learning-techniques/
'''

import json
from pickletools import optimize
from matplotlib import units
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATASET_PATH = "./data_with_labels.json"

# Load data
def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
        
    # Convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    
    return inputs, targets


if __name__ == "__main__":
    # Load data
    inputs, targets = load_data(DATASET_PATH)
    
    # Split the data into train and test set
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)
     
    # Build the model architecture using Tenserflow
    model = keras.Sequential([
        keras.layers.Flatten(input_shape = (inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(units = 512, activation = "relu"),
        keras.layers.Dense(units = 256, activation = "relu"),
        keras.layers.Dense(units = 64, activation = "relu"),
        keras.layers.Dense(units = 10, activation = "softmax")
        ])

    # Compile network
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer=optimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics=["accuracy"]
                  )

    model.summary()
    
    # Train network
    model.fit(inputs_train, 
              targets_train, 
              validation_data=(inputs_test, targets_test),
              epochs=50,
              batch_size=32
              )

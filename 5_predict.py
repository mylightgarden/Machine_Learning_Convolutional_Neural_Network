''' 
Reference : https://www.youtube.com/c/ValerioVelardoTheSoundofAI
https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8?gi=ac21c958a822
https://www.analyticsvidhya.com/blog/2021/06/music-genres-classification-using-deep-learning-techniques/
'''

from ast import operator
import json
from pickletools import optimize
from matplotlib import units
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import librosa
import os

SAMPLE_RATE = 22050
TRACK_DURATION = 30 # each audio record is 30 second
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

genres = ['blue', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
   
# Load data
def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
        
    # Convert lists into numpy arrays
    inputs = np.array(data["mfcc"])          
    targets = np.array(data["labels"])       
    
    return inputs, targets

def predict(X):
    result = [None] * 10
     
    new_model = tf.keras.models.load_model(r'C:\Users\Administrator\Desktop\Oregon\MusicGenre\GTZAN_Genre\final\saved_model\my_model')
    
    # model.predict expect a 4 dementional array
    # x is currently 3 dementional like (130,13,1)
    # We need to change it to (1,130,13,1)     
    X = X[np.newaxis, ...] 
    
   # The result of the prediction will be a two dimentional array like [[0.1, 0,2, ...]]  # print(" The predition is :", prediction)
    prediction = new_model.predict(X)
    prediction_0 = prediction[0]
    
    percentage_list = []
    for i in range(len(prediction_0)):
        percentage_list.append("{0:.2%}".format(prediction_0[i]))
        
    my_result = list(zip(genres, percentage_list))
    my_result_sorted = sorted(my_result, key = lambda x: x[1])
    
    print("---------Predicting Genres-------------")
    for i in range(len(my_result)-1, -1, -1):
        print(my_result_sorted[i])
       
    # To get the top 1: extract index with max value    
    predicted_index = np.argmax(prediction, axis=1) # will get the label number e.g. : [4]
    # print("The highest predicted index is {}".format(predicted_index))

if __name__ == "__main__":    
    # Using librosa to load audio file
    dir_path = "C:/Users/Administrator/Desktop/Oregon/MusicGenre/GTZAN_Genre/new_music/Mozart-Serenade-in-G-major.au"
    #f = "/blues.00001.au"
    #file_path = os.path.join(dir_path, f)
    #file_path = input_audio_path
    signal, sample_rate = librosa.load(dir_path, sr=SAMPLE_RATE)
    y_l = int(SAMPLES_PER_TRACK/10)
    mfcc = librosa.feature.mfcc(y=signal[y_l:y_l*2], sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    #data = mfcc.tolist()
    mfcc = mfcc[..., np.newaxis]
    #print("This is mfcc: ", mfcc)
    predict(mfcc)
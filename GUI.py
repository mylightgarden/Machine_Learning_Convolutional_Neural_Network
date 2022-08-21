# ---------- Graphical User Interface ----------
from cProfile import label
from multiprocessing.connection import wait
import tkinter as tk
import librosa
import librosa.display
import matplotlib.pyplot as plt
import cv2
from saved_model import *
from tkinter import filedialog
from tkinter import messagebox as tkMessageBox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json
from pickletools import optimize
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import os


SAMPLE_RATE = 22050
TRACK_DURATION = 30 # each audio record is 30 second
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION
dirname = os.path.dirname(__file__)

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
    model_path = os.path.join(dirname, 'saved_model/my_model') 
    new_model = tf.keras.models.load_model(model_path)
    
    # model.predict expect a 4 dementional array
    # x is currently 3 dementional like (130,13,1)
    # We need to change it to (1,130,13,1)     
    X = X[np.newaxis, ...] 
    
   # The result of the prediction will be a two dimentional array like [[0.1, 0,2, ...]]  
   # # print(" The predition is :", prediction)
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
        
    return prediction[0]

# ---------- Variables ----------
genre_map = {0: 'Blues', 1: 'Classical', 2: 'Country', 3:'Disco', 4: 'Hiphop', 5: 'Jazz', 6: 'Metal', 7: 'Pop', 8: 'Reggae', 9: 'Rock'}
file_path = ''


# ---------- Functions ----------
def select():
    global file_path
    file_path = filedialog.askopenfilename(initialdir = "/", title = "Select an audio file", filetypes=[("Audio Files", ".wav .au .mp3")])

    # Check if select a file
    if file_path:
        # Show the go button
        top2_path = os.path.join(dirname, 'Images/top_2.png') 
        go_image = tk.PhotoImage(file=top2_path)
        go_button.configure(image=go_image)
        go_button.image = go_image
        
        # Show the name of the selected file
        file_name = file_path.split('/')[-1]
        entry1_text.set(file_name)
    else:
        tkMessageBox.showerror(title="Error", message="You have not chosen the audio file. Please select!")

def go():
    # Clean frame: div1
    for widget in div1.winfo_children():
        widget.destroy()

    # Check the existence of the file_path
    if not file_path:
        tkMessageBox.showerror(title="Error", message="You have not chosen the audio file. Please select!")
        return

    # Data Preprocessing -> MFCCs -> Spectrogram Image
    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)
    y_l = int(SAMPLES_PER_TRACK/10)
    mfcc = librosa.feature.mfcc(y=signal[y_l:y_l*2], sr=sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
    mfcc = mfcc.T
    mfcc = mfcc[..., np.newaxis]
    #print("This is mfcc: ", mfcc)
    prob = predict(mfcc)

    # Get the result
    result = {}
    for num in range(10):
        # Genre: probability
        result[genre_map[num]] = prob[num]

    # Sort in descending order
    sorted_result = sorted(result.items(), key=lambda x: x[1])

    # Store the results
    genres = []
    probabilities = []
    for res in sorted_result:
        genres.append(res[0])
        probabilities.append(float("{:.2f}".format(res[1]*100)))

    # Show the figure result
    fig = plt.figure()
    plt.barh(genres, probabilities)
    plt.title("Most Likely Music Genre")
    plt.xlabel('Probability(%)')
    for i, val in enumerate(probabilities):
        plt.text(val + 3, i, str(val)+'%', color = 'black', fontweight = 'bold')
    canvas = FigureCanvasTkAgg(fig, master=div1)
    canvas.get_tk_widget().pack()
    plt.close()

    # Reset
    go_button.configure(image=button_image)
    go_button.image = button_image
    entry1_text.set('')

def quit_message():
    res = tkMessageBox.askyesno(title="Exit", message="Are you sure you want to exit?")

    # If receiving yes, then leave the program
    if res:
        window.quit()


# ---------- GUI ----------
window = tk.Tk()
window.title("CS 467 Online Capstone Project")

label1 = tk.Label(window, text = "Selected Audio File: ", font=("Futura", 16))
label1.grid(row = 0, column = 0)

entry1_text = tk.StringVar()
entry1 = tk.Entry(window, textvariable=entry1_text)
entry1.grid(row = 0, column = 1)

# Button to select the audio file
select_button = tk.Button(window, text='Select', command=select)
select_button.grid(row = 0, column = 2)

# Button to show the instructions & start predicting
top1_path = os.path.join(dirname, 'Images/top_1.png') 
button_image = tk.PhotoImage(file=top1_path)
go_button = tk.Button(window, image=button_image, command=go)
go_button.grid(row = 1, column = 0, columnspan=3)

# Division for results
div1 = tk.Frame(window,  width=600, height=450, bg='white')
div1.grid(row = 2, column = 0, columnspan=3)

# Exit message
window.protocol("WM_DELETE_WINDOW", quit_message)

window.mainloop()

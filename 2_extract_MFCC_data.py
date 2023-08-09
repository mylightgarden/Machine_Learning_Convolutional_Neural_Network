# Author: Sophie Zhao

''' 
Reference : https://www.youtube.com/c/ValerioVelardoTheSoundofAI
https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8?gi=ac21c958a822
https://www.analyticsvidhya.com/blog/2021/06/music-genres-classification-using-deep-learning-techniques/
'''

import os
import math
import librosa
import json

DATASET_PATH = "../genres"
JSON_PATH = "./data_with_labels.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # each audio record is 30 second
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION


# Purpose: Extracts MFCCs from music dataset and saves them into a json file along witgh genre labels
def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5): 

    # dictionary to store mapping, labels, and MFCCs
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []        
    }
    
    # because our dataset is relatively small, we use segment to divid one recored to multiple segments.
    num_sample_per_seg = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_sample_per_seg / hop_length)

    # loop through all genre sub-folder
    # dirpath is the current folder
    # dirnames are all the subfolders
    # filenames are all the files in the subfolders.
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # Check we are not at root level
        if dirpath is not dataset_path:

            # parse the folder directory and save sub-folder name in the mapping. e.g. blue
            dirpath_components = dirpath.split("\\")
            semantic_label = dirpath_components[-1]
            #print("--------",semantic_label)
            data["mapping"].append(semantic_label)
            print("\n-------------------------Processing: {}--------------------------".format(semantic_label))

            # process all audio files
            for f in filenames:

		        # Using librosa to load audio file
                file_path = os.path.join(dirpath, f)
                signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                # For each audio track, divid it into segment, and process each segment
                for s in range(num_segments):

                    seg_start = num_sample_per_seg * s
                    seg_end = seg_start + num_sample_per_seg            

                    # extract mfcc
                    mfcc = librosa.feature.mfcc(y=signal[seg_start:seg_end], 
                                                sr=sample_rate, 
                                                n_mfcc=num_mfcc, 
                                                n_fft=n_fft, 
                                                hop_length=hop_length)
                    mfcc = mfcc.T

                    # store only mfcc feature with expected number of vectors
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        # The first round of i was for the dataset_path, so we need to -1
                        data["labels"].append(i-1)
                        print("{}, segment:{}".format(file_path, s+1))

    # save MFCCs to json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
        
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)

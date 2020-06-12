#!/usr/bin/env python
# coding: utf-8

# In[8]:


import json
import os
import math
import librosa
from json import JSONEncoder


DATASET_PATH = "untitled.wav"
JSON_PATH = "newfile.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": []
    }
    
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)
    signal, sample_rate = librosa.load(dataset_path, sr=SAMPLE_RATE)
    for d in range(num_segments):
        # calculate start and finish sample for current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment
        # extract mfcc
        mfcc = librosa.feature.mfcc(signal[start:finish], sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            #print("{}, segment:{}".format(file_path, d+1))
            
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
    

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=10)


# In[1]:


import tensorflow as tf
import numpy as np
import json
model=tf.keras.models.load_model('trainedmodel.h5')


# In[2]:


with open("singlefileprocessing.json","r") as fp:
    predictionarray=json.load(fp)
newarray = np.array(predictionarray["mfcc"])
#print(newarray[5].shape)
newarray1=newarray[5]
#print(newarray1)
#print(newarray1.shape)
newarray2 = newarray1[..., np.newaxis]
#print(newarray2.shape)
newarray3=newarray2[np.newaxis,...]
#print(newarray3.shape)
predict23=model.predict(newarray3)
predictnew = np.argmax(predict23, axis=1)
print(predictnew)



# %%

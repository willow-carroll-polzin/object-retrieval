'''
SYSC 5906:
Object Retrieval System - V1.0
---
Script to deploy three networks to peform a object retrieval task.

The main components of the system are:
1.   The first network conists of a object detection network based on YOLACT that takes in RGBD frames and outputs ???
2.   The second network takes ???? and outputs labels representing the type of
room that the objects likely belong to. The current output of the network
is recorded and a list of all detected rooms and their corresponding objects is
retained.
3.   The third network takes in object label from a user and and the list of detected rooms and outputs a softmax representing the most likely room that the desired object would be located in.

This softmax and room label are then used to create a route to the desired room. 

Then ???????

NOTE: This version of the system runs on **singular frames** from videos
'''
########
# SETUP:
########
#Directories with model weights and datasets
NN1_OD_DIRECTORY = '/gdrive/My Drive/Colab Notebooks/SYSC 5906/models/weights/V2 Improved versions/object_detector/'   #Object detector
NN2_RD_DIRECTORY = '/gdrive/My Drive/Colab Notebooks/SYSC 5906/models/weights/V2 Improved versions/room_classifier_3/' #Room detector/guessor
TEST_DATASET = '/gdrive/My Drive/Colab Notebooks/SYSC 5906/datasets/mit_indoors/processed/data_labelsOnly/'
PICKLE_DIRECTORY = '/gdrive/My Drive/Colab Notebooks/SYSC 5906/datasets/mit_indoors/processed/data_labelsOnly/'

#Import libraries and scripts
import tensorflow as tf
import pandas as pd
import pickle
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import keras.api._v2.keras as keras
from keras.models import load_model
from sklearn.model_selection import train_test_split

from room_detection import roomGuesser, roomDetector
from object_detection import objectDetector

#Import dataset
pickledData = open(PICKLE_DIRECTORY+"listOfAllObj_v3.pkl","rb")
dataSet = pickle.load(pickledData)
pickledData.close()

#Import list of unqiue objects from training dataset
pickledObjs = open(PICKLE_DIRECTORY+"uniqueObjs_v3.pkl","rb")
uniqueObjs = pickle.load(pickledObjs)
uniqueObjs = dataSet.columns[0:-1]
pickledObjs.close()

#Load models
model_1_OD = tf.keras.models.load_model(NN1_OD_DIRECTORY)
model_2_RD = tf.keras.models.load_model(NN2_RD_DIRECTORY)

#Summarize models
model_1_OD.summary()
model_2_RD.summary()

########
# MAIN LOOP:
########
#Get a desired object from the user
targetObj = input('Enter the desired object: ')
#print(f'You entered - {targetObj}')

targetObjStatus = False
while(targetObjStatus == False):
    ########
    # OBJECT DETECTION:
    ########
    #Detect objects in current frame
    detectedObjects, cameraPose = objectDetector(model_1_OD)

    ########
    # TARGET CHECK:
    ########
    if targetObj in detectedObjects:
        targetObjStatus = True
        print(f'The {targetObj} has been found!' )
        break
    else:
        print('Still searching for target object')

    ########
    # ROOM DETECTION:
    ########
    #Label rooms based on currently detected objects
    detectedRooms, = roomDetector(detectedObjects,model_2_RD)

    ########
    # ROOM GUESSING:
    ########
    #Search for singular objects in the current input and return top results
    result, n = roomGuesser(targetObj, uniqueObjs, model_2_RD)

    #Check if result is valid
    if n == 0:
        print(result[0])
        #Get a new desired object from the user
        targetObj = input('Enter a valid desired object: ')
        break

    ########
    # PATH PLANNING:
    ########

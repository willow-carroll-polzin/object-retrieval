'''
SYSC 5906:
Semantic Mapping System - V1.0
---
This system performs semantic mapping tasks of a indoor space.

The main components of the system are:
1.   Metric map generation using RGBD data
2.   Object detection performed on live feed from RGB data
3.   Room detection based on object labels from object detection
4.   Labelling of the metric map with room labels
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
import pyrealsense2 as rs
import math as m
from keras.models import load_model
from sklearn.model_selection import train_test_split

from room_detection.room_detection import roomGuesser, roomDetector
from object_detection import objectDetector
from map_generation import label_map

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






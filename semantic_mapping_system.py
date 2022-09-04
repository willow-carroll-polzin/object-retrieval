'''
SYSC 5906:
Semantic Mapping System - V1.0
---
This system performs semantic mapping tasks of a indoor space.

The main components of the system are:
1.   Metric map generation using live or prerecorded video with RGBD data
2.   Object detection performed on live or prerecorded video with RGB data
3.   Room detection based on object labels from object detection
4.   Labelling of the metric map with room labels and camera pose from live or prerecorded video
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
import math as m'
from keras.models import load_model
from sklearn.model_selection import train_test_split

from room_detection.room_detection import roomDetector
from object_detection.object_detection import objectDetector
from object_detection.vision_system import cameraSetup, openPoses
from map_generation.map_generation import labelPath, filterPath

#Load custom models
model_RD = tf.keras.models.load_model(NN2_RD_DIRECTORY)

#Summarize models
model_RD.summary()

####################################
# OFFLINE VERSION
# This version of the system loads a 
# pre-recorded map and video file.
####################################
########
# ACCESS PRE-RECORDED DATA (VIDEO+POSES):
########
OFFLINE = True
frames = cameraSetup(OFFLINE)
poses = openPoses()

########
# MAIN LOOP:
########
for currentFrame in frames:
    ########
    # OBJECT DETECTION:
    ########
    #Detect objects in current frame
    detectedObjects = objectDetector(currentFrame)

    ########
    # ROOM DETECTION:
    ########
    #Label rooms based on currently detected objects
    detectedRooms = roomDetector(detectedObjects,model_RD)

    ########
    # MAPPING:
    ########
    #Parse the pose data and append room labels to each pose.
    #Each pose corresponds to a singular frame, and therefore
    #A singular room label
    labeledPath = labelPath(detectedRooms, poses)

    #Apply a low-pass filter to the list which contains the
    #labelled path to determine the aproximate centroid of each room
    #Each entry in the list consists of x,y,room_label data
    filterPath(labeledPath)






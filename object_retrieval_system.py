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
from room_detection import roomGuesser
from keras.models import load_model
from sklearn.model_selection import train_test_split

#Load models
model_1_OD = tf.keras.models.load_model(NN1_OD_DIRECTORY)
model_2_RD = tf.keras.models.load_model(NN2_RD_DIRECTORY)

#Summarize models
model_1_OD.summary()
model_2_RD.summary()

########
# OBJECT DETECTION:
########

########
# ROOM DETECTION:
########

########
# ROOM GUESSING:
########

########
# PATH PLANNING:
########

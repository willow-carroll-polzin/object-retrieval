'''
SYSC 5906:
Object Retrieval System - V1.0
---
This system peforms a object retrieval task inside a known semantically mapped indoor space.

The main components of the system are:
1.   Guessing the most likely room(s) that a object will be located in
2.   Path planning to the likely room(s)
3.   Searching of the space untill the desired object is detected
'''
########
# SETUP:
########
#Directories with model weights and datasets

NN_RD_DIRECTORY = './models/nn_room_detector/' #Room detector/guessor
TEST_DATASET = '/gdrive/My Drive/Colab Notebooks/SYSC 5906/datasets/mit_indoors/processed/data_labelsOnly/'
RD_TRAINED_DATA_DIRECTORY = '/models/trained_data/data_labelsOnly/'
MAP_PATH = '/datasets/maps/sample/'

#Import libraries and scripts
import tensorflow as tf
import pandas as pd
import pickle
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import keras.api._v2.keras as keras
from keras.models import load_model
from sklearn.model_selection import train_test_split

from room_detection.room_detection import roomGuesser, roomDetector
from object_detection.object_detection import objectDetector
from object_detection.vision_system import cameraSetup
from path_planner.path_planner import path_planner
from object_detection.yolact.eval import setup, evalFrame
from semantic_mapping_system import map,convertCOCO2MIT_tensor



#Import dataset
# pickledData = open(RD_TRAINED_DATA_DIRECTORY+"listOfAllObj_v3.pkl","rb")
# dataSet = pickle.load(pickledData)
# pickledData.close()

#Import list of unqiue objects from training dataset
# pickledObjs = open(RD_TRAINED_DATA_DIRECTORY+"uniqueObjs_v3.pkl","rb")
# uniqueObjs = pickle.load(pickledObjs)
# uniqueObjs = dataSet.columns[0:-1]
# pickledObjs.close()

# #Load custom models
# model_OD = tf.keras.models.load_model(NN1_OD_DIRECTORY)
with tf.device('/cpu:0'):
    model_RD = tf.keras.models.load_model(NN_RD_DIRECTORY)

# #Summarize models
# model_OD.summary()
# model_RD.summary()




MIT_CLASSES = open("./object_detection/yolact/data/archive/mit_data.txt","r").read().split("\n")
MIT_CLASSES = MIT_CLASSES[1:]

####################################
# OFFLINE VERSION
# This version of the system loads a 
# pre-recorded map and video file.
####################################
########

# ROOM GUESSING:
########
#Get a desired object from the user
targetObj = input('Enter the desired object: ')

# #Search for singular objects in the current input and return top results
# result, n = roomGuesser(targetObj, MIT_CLASSES, model_RD)
# del model_RD, result, n

#Check if result is valid
# if n == 0:
#     print(result[0])
#     #Get a new desired object from the user
#     targetObj = input('Enter a valid desired object: ')

########
# INTIAL PATH PLAN:
########
# get current pose
# plan path from current pose

########
# ACCESS PRE-RECORDED DATA (VIDEO+POSES):
########
OFFLINE = True
frames = cameraSetup(OFFLINE)

########
# MAIN LOOP:
########




targetObjStatus = False
haveFrames = True
# OBJECT DETECTION:
########
torch.cuda.empty_cache()
net, dataset, class_names,label_map = setup()
m=map(MIT_CLASSES,class_names,label_map)
########
while(not(targetObjStatus)):


    for currentFrame in frames:
        #Check if target was found in the last analysed frame
        if targetObjStatus:
            break
        ########
        # OBJECT DETECTION:
        ########
        #Detect objects in current frame
        obj_tensor_coco,img = evalFrame(net,currentFrame)
        obj_name_list = []
        for i in range(len(obj_tensor_coco)):
            if obj_tensor_coco[i]>0:
                obj_name_list.append(class_names[i])

        ########
        # TARGET CHECK:
        ########
        if targetObj in obj_name_list:
            targetObjStatus = True
            print(f'The {targetObj} has been found!' )
            cv2.imwrite("result.jpg",img)
            correct = input("Has the target been correctly found? (y/n)")
            if(correct != "y"):
                targetObjStatus = False
                print("Continuing to search...")
        else:
            print('Still searching for target object')
        
        #switch active model
        #torch.cuda.empty_cache()
        #model_RD = tf.keras.models.load_model(NN_RD_DIRECTORY)
        ########
        # ROOM DETECTION:
        ########
        # #Label rooms based on currently detected objects
        # mit_object = tf.constant(convertCOCO2MIT_tensor(obj_tensor,m),shape=(1,2204))
        # detectedRooms = roomDetector(mit_object,model_RD)
        # del obj_tensor
        #model_RD = model_RD.cpu()
        ########
        # PATH PLANNING:
        ########
        #Load in pre-recorded map

        #Load in pre-recorded localized rooms
        #rooms = grab_rooms()

        #Get current pose for the frame

with tf.device('/cpu:0'):
    model_RD = tf.keras.models.load_model(NN_RD_DIRECTORY)
# #Search for singular objects in the current input and return top results
mit_object = tf.constant(convertCOCO2MIT_tensor(obj_tensor_coco,m),shape=(1,2204))
detectedRooms,roomStr = roomDetector(mit_object,model_RD)

result, n = roomGuesser(targetObj, MIT_CLASSES, model_RD)

print(targetObj+ " was found in the " + roomStr)
cv2.imwrite("result.jpg",img)


        #path_planner(MAP_PATH, targetObjStatus, detectedRooms, rooms, current_pose)
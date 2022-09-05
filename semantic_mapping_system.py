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
NN_RD_DIRECTORY = './models/nn_room_detector/' #Room detector/guessor
TEST_DATASET = '/gdrive/My Drive/Colab Notebooks/SYSC 5906/datasets/mit_indoors/processed/data_labelsOnly/'
PICKLE_DIRECTORY = '/models/trained_data/data_labelsOnly/'
DETECTED_OBJS_DIRECTORY = '/dataset/detectedObjs'

#Import libraries and scripts
import tensorflow as tf
import pandas as pd
import pickle
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np
import glob
#import tensorflow_datasets as tfds
import keras.api._v2.keras as keras
import pyrealsense2 as rs
import math as m
from keras.models import load_model
from sklearn.model_selection import train_test_split

from room_detection.room_detection import roomDetector
from object_detection.object_detection import objectDetector
from object_detection.vision_system import cameraSetup
from map_generation.map_generation import labelPath, roomLocalizer
from object_detection.yolact.eval import setup, evalFrame



####################################
# OFFLINE VERSION
# This version of the system loads a 
# pre-recorded map and video file.
####################################

########
# SETUP YOLACT MODEL
########
net, dataset, class_names,label_map = setup()
coco_class_list = class_names
COCO_MAP = label_map
print(coco_class_list)
########
# ACCESS PRE-RECORDED DATA (VIDEO+POSES):
########
OFFLINE = True
frames= cameraSetup(OFFLINE)
print(str(len(frames)) + " frames found.") 

MIT_CLASSES = open("./object_detection/yolact/data/archive/mit_data.txt","r").read().split("\n")
MIT_CLASSES = MIT_CLASSES[1:]
def map(mit,coco,m):
    temp_map=[]
    i = 0
    for coco_class in coco:
        name = coco_class
        #print(coco_class)
        #print(coco)
        coco_index = i
        #print(coco_index)
        coco_mapped = list(m.keys())[coco_index]
        if coco_class in mit:
            mit_index = mit.index(coco_class)
        else:
            mit_index = None
        temp_map.append({"name":name,"coco_index":coco_index,"coco_mapped":coco_mapped,"mit_index":mit_index})
        i=i+1
    return temp_map

def convertCOCO2MIT_tensor(coco_obj,map):
    mit_obj = np.zeros(len(MIT_CLASSES))
    for entry in map:
        if entry["mit_index"] is not None:
            #print(entry)
            mit_obj[entry["mit_index"]]=coco_obj[entry["coco_index"]]
            #print(entry["mit_index"])
            #print(MIT_CLASSES[entry["mit_index"]])
    return mit_obj

test_coco_obj = np.zeros(len(coco_class_list))
target = "cup"
print(target in coco_class_list)
test_coco_obj[coco_class_list.index(target)]=1
print("COCO: "+target)
print(test_coco_obj)
m = map(MIT_CLASSES,coco_class_list,COCO_MAP)
test_mit_obj = convertCOCO2MIT_tensor(test_coco_obj,m)
print(test_mit_obj)
print(MIT_CLASSES[np.where(test_mit_obj != 0)[0][0]])

########
# MAIN LOOP:
########
obj_tensors = []
for currentFrame in frames:
    #print(currentFrame)
    ########
    # OBJECT DETECTION:
    ########
    #Detect objects in current frame
    obj_tensors.append(evalFrame(net,currentFrame))
    #print(obj_tensor)
    #Get detected objects from PKL file
    # pickledObjs = open(DETECTED_OBJS_DIRECTORY+".pkl","rb")
    # detectedObjects = pickle.load(pickledObjs)
    # detectedObjects = dataSet.columns[0:-1]
    # pickledObjs.close()

del net

#Load custom models
model_RD = tf.keras.models.load_model(NN_RD_DIRECTORY)

#Summarize models
model_RD.summary()
for obj_tensor in obj_tensors:
    #Convert coco object list tensor to mit label object list tensor
    mit_object = tf.constant(convertCOCO2MIT_tensor(obj_tensor,m),shape=(1,2204))

    ########
    # ROOM DETECTION:
    ########
    #Label rooms based on currently detected objects
    detectedRooms = roomDetector(mit_object,model_RD)
    # print(detectedRooms)
    ########
    # MAPPING:
    ########
    #Parse the pose data and append room labels to each pose.
    #Each pose corresponds to a singular frame, and therefore
    #A singular room label
    # labeledPath = labelPath(detectedRooms, path)

    #???????
    #rooms = roomLocalizer(labeledPath)

    #Save rooms as data??


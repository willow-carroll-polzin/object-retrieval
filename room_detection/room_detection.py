import pandas as pd
import numpy as np
import tensorflow as tf

"""
This method uses a custom FNN to detect rooms based on object labels. These
object labels are gather by another network performing object detection on a 
segmented RGBD image.

Note: This script must be located in the room_classifier folder which contains the 
model as a .pb file.
"""
def roomDetector(x_val_tensor, model):
    #Wrap the model so it returns a probability
    probability_model = tf.keras.Sequential([
        model,
        tf.keras.layers.Softmax()
    ])

    #Run room detection model
    y_pred = model.predict(x_val_tensor)

    #Determine classification from the outputed probabilities
    y_classes = []
    for y in y_pred:
        y_classes.append(y.argmax())
    print(y_classes)

    return y_classes

"""
This method uses a custom FNN which detects room to detect the most likely 
room that a desired object can be found in. This is achieved by calling the model
with only a single object as a input (rather then a list of detected objects).

Note: This script must be located in the room_classifier folder which contains the 
model as a .pb file.
"""
def roomGuesser(targetObj, uniqueObjs, model):
    #Get list of unique object names from dataset (column titles EXCEPT room name)
    objectList = pd.DataFrame(uniqueObjs)
    #Check that user inputed desired object (target) is valid
    if targetObj not in objectList.values:
        error = ["Invalid input object"]
        return error, 0

    #Init empty vector that is the same size of unique objects list
    oneHotSearchInput = np.zeros(len(objectList))

    #Find the number that corresponds to the object label/string
    location = objectList[objectList.eq(targetObj).any(1)].index.values[0]

    #Potential rooms, used to map from number to room label/string
    desiredRooms = ['bathroom','bedroom','dining_room','corridor','livingroom','kitchen','office']

    #Set input to true at index of target object
    oneHotSearchInput[location] = 1

    #Convert vector to tensor
    oneHotTensor = tf.constant(oneHotSearchInput,shape=(1,len(uniqueObjs)))

    #Predict romm based on tensor of objects
    target_pred = model.predict(oneHotTensor)

    #Sort results into a tuple (links model probabilities with room labels/strings)
    results = []
    for room in range(len(desiredRooms)):
        results.append((target_pred[0][room],desiredRooms[room]))
    sorted_results = sorted(results,key=lambda x : x[0])

    #Find the top n most likely results
    n=3
    top_n = sorted_results[-n:]

    #Print the top n results
    print("Target: " + targetObj + " will most likely be found in:")
    for i in range(1,n+1,1):
        print(" "+top_n[-i][1] + " (" + str(top_n[-i][0]) + ")")

    return top_n, n
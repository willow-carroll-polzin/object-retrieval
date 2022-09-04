import pandas as pd
import numpy as np
import tensorflow as tf

"""
This method uses YOLACT trained on the COCO dataset to detect objects
from RGB camera frames.
"""
def objectDetector(image, model):
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
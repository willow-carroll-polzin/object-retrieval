import glob
from PIL import Image
import os
import pandas as pd


def openImages():
    image_list = []
    filelist = glob.glob('/dataset/images/*.jpg')
    filelist = os.listdir('./dataset/images/sample/')
    for file in filelist:
        image = Image.open(os.getcwd()+"/dataset/images/sample/"+file)
        image_list.append(os.getcwd()+"/dataset/images/sample/"+file)
    return image_list

def openPath():
    posesDF = pd.read_csv('/dataset/poses/*.csv')
    print(df.head())
    return posesDF
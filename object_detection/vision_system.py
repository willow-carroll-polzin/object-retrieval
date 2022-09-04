import pyrealsense2 as rs
import pandas as pd
import math as m
import glob
from PIL import image


def openImages():
    image_list = []
    for file in glob.glob('dataset/iamges/*.jpg'):
        image = Image.open(file)
        image_list.appen(image)
    return image_list

def openPoses():
    posesDF = pd.read_csv('dataset/iamges/*.csv')
    print(df.head())
    return posesDF

def cameraSetup(OFFLINE):
    if OFFLINE:
        return openImages()
    else:
        print("ONLINE MODE - TURN RS CAMS ON")

# pipe_pose = rs.pipeline()
# cfg_pose = rs.config()
# cfg_pose.enable_stream(rs.stream.pose)
# pipe_pose.start(cfg_pose)

# pipe_clr = rs.pipeline()
# cfg_clr = rs.config()
# cfg_clr.enable_stream(rs.stream.color)
# pipe_clr.start(cfg_clr)

# try:
#     while (True):
#         frames_pose = pipe_pose.wait_for_frames()
#         pose_frame = frames_pose.get_pose_frame()
#         pose_data = pose_frame.get_pose_data()
#         x = -pose_data.rotation.z
#         y = pose_data.rotation.x

#         frames_clr = pipe_clr.wait_for_frames()
#         clr_frame = frames_clr.get_color_frame()
#         clr_data = clr_frame.get_color_data()
# finally:
#     pipe_pose.stop()



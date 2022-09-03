# SYSC 5906: DIRECTED STUDY
## Room Recognition
### Team Members
Max Polzin  
Keyanna Coghlan  
Sami Nassif-Lachapelle  

### Overview
This project uses deep learning models to generate semantically labelled maps of indoor spaces. 

The system consists of three main components:
1. A object detector using version of YOLACT trained on the MIT Indoor Scenes Dataset (2019)
2. A custom "room detector" that takes in detected objects and their position and infers the room label
3. A custom path planner/obstacle avoidance system that generates a metric map of the environment and learns to avoid walls and obstacles
4. Finally the output of the path planner and obstacle avoidance are combined to generate a semantically labelled map of the environment
![alt text](https://github.com/MaxPolzinCU/room-recognition/images/systemOverview.png?raw=true)


FINISH UPDATING THE REST OF THE README:

## Setup and Usage:
### Setting up the Environment and Dependencies
1. Unzip the provided file called "SemanticMapLabels.zip" OR visit https://github.com/MaxPolzinCU/SemanticMapLabels and clone the repository
2. Install Python 3.8.5 and Pip 20.0.2 or greater
3. Install the Intel RealSense SDK Python wrapper with the following command: "pip install pyrealsense2"
4. Install the following libraries:
- numpy 1.20.2
- matplotlib 3.4.1
- openCV 4.5.2
5. Clone the official Darknet YOLOv3 repository found at: https://github.com/pjreddie/darknet

Follow the instructions provided in the following link to build the library: https://pjreddie.com/darknet/yolo/
- The MakeFile parameters used are all default except for as follows:
    - GPU=1
    - CUDNN=0
    - OPENCV=1
    - OPENMP=0
    - DEBUG=0
Once completed the desired weights (e.g. "yolov3-tiny.weights") should be moved into the ./model/weights folder and the compiled library "libdarknet.so" should be placed in the ./model folder. 

Alternatly the weights provided in this repository and the compiled "libdarknet.so" can be used if this code is being run on a CUDA enabled GPU. Note this will only work on a Unix machine, as "libdarknet.so" wont run on Windows.

### Running the system
1. Open a terminal and cd into the "SemanticMapLabels" folder.
2. Run: "python classifierDepth.py" \
This will generate a 2D plot representing the environment captured by the stereo camera with annotated labels of the detected objects.   
4. Run: "python classifierWebcam2.py" \
This will access the computers webcam, if available, and perform objection detection while also ????????

## Repo Contents:
- **classifierDepth.py**: Main script needed, allows for both object detection and mapping using the Inteal RealSense D435i stereo camera

- **classifierSingle.py**: Test object detection on a single image, feed it a input image from ./models/data

- **classifierWebcam.py**: Test object detection with a webcam

- **classifierWebcam2.py**: Test object detection with a webcam, uses a structure more similar to that of the main "classifierDepth.py" script

- **models**: Folder to contain everything related to the ML models
    - *weights*: Folder to contain pre-trained weights for YOLOv3 Network
        - yolov3-tiny.weights: Pre-trained wegihts
    - *data*: Folder to contain and labels or datasets to be used
        - dog.jpg: Test image for "classifierSingle.py"
        - coco.names: Contains all the labels from the COCO Dataset
    - *cfg*: Folder to contain all config files for networks used
        - coco.data: Config paramters for COCO Dataset
    - *libdarknet.so*: Pre-compiled Darknet library using YOLOv3 and trained on ????


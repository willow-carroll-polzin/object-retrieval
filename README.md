# SYSC 5906: DIRECTED STUDY
## Room Recognition
### Team Members
Max Polzin  
Keyanna Coghlan  
Sami Nassif-Lachapelle  

### Overview
This project uses deep learning models to search for objects inside indoor spaces.

The system consists of three main components:
1. A object detector using version of YOLACT
2. A custom "room detector" that takes in detected objects infers the room labels for the current position of the robot
3. A filter and mappings system that better defines the boundaries of each room
4. A path planner that creates a route to the most likely room that a desired object may be found in

### System Diagram
![alt text](https://github.com/MaxPolzinCU/room-recognition/blob/main/images/systemOverview.png)

## Setup and Usage:
### Setting up the Environment and Dependencies
1. Clone this repository
2. Install ROS 20.04 Noetic
3. Install Python 3.8.5 and Pip 20.0.2 or greater
4. Install the Intel RealSense SDK Python wrapper with the following command: "pip install pyrealsense2"
5. Install the Intel RealSense ROS wrapper

## Running the system
### Step 1: Generate a map
1. Open a terminal and cd into the "support_scripts/ros_scripts" folder.
2. Run: "bash run_cameras.sh" \
This will allow the cameras to start collecting data to generate a map.   
4. Stop the first bash script and run: "bash run_rosbag.sh && export_data.sh" \
This will bag the data and export the pose data. Make sure the data is then stored in the relevant folders under "/dataset/" such as "images", "maps", and "poses". \

### Step 2: Determine the current room
1. Open a terminal
2. Run: "semantic_mapping_system.py" \
This will return the current room for each frame found in "dataset/images". The functions in "map_generation.py" are used to correlate
each detected room for each frame to aproximate locations based on the pose data from the cameras, which is found in "dataset/poses".

### Step 3: Retrieve a object
1. Open a terminal
2: Run: "object_retrieval_system.py" \
This will prompt you for a object label, type your desired object in the terminal (e.g. "cup"). If the object is detected in the collected images
the frame with the object will be displayed. Another prompt will be given in the terminal, respond with yes or no to let the system know if the 
correct object was detected. \
Once the correct object has been found a path with be plotted using the functions in "path_planner.py" and the path will be displayed on top of the map
that is found in "dataset/map".

## Repo Contents:
- **map_generation**: Folder containing .py files needed for the mapping tasks.

- **object_detection**: Folder containing .py files needed for the object detection tasks. Additionally the files and weights used for YOLACT are stored in a subfolder here.

- **path_planner**: Folder containing .py files needed for the path planning tasks.

- **room_detection**: Folder containing .py files which define the functions needed for the room detection and estimation tasks.

- **support_scripts**: Folder containing bash scripts used to execute ROS code to capture data as well as .py files with functions used in other parts of the system.

- **dataset**: Folder containing sample data to test the system, this is also where collected data should be stored when using the system.

- **models**: Folder containing everything related to the custom DL models used in this system.
    - *data_processors*: Folder to scripts to clean and modify raw data used to train the custom ML models (e.g. convert MIT dataset to a usable format)
    - *nn_room_detector*: Folder containing the weights and architecture of the custom room detector neural network.
    - *trained_data*: Folder containing .pkl files relating to the datasets used to train the custom room detector, such as a list of unique objects that the detector can take as inputs.
    - Various .ipynb notebook files used to define, train, and validate the custom DL models used in this system.


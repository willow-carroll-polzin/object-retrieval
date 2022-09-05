from PIL import Image
from numpy import asarray
import numpy as np
import random as rand

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

def astar(map, start, end):
    # Create start and end node
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    # Initialize both open and closed list
    open_list = []
    closed_list = []

    # Add the start node
    open_list.append(start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        # Pop current off open list, add to closed list
        open_list.pop(current_index)
        closed_list.append(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1] # Return reversed path

        # Generate children
        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(map) - 1) or node_position[0] < 0 or node_position[1] > (len(map[len(map)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if map[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            open_list.append(child)

def path_planner(map_path, obj, p_room, rooms, current_pose):
  # Open map image
  map_image = Image.open(map_path)

  # Convert map image to array
  raw_map = asarray(map_image)

  # C
  map = [[0 for i in range(len(raw_map[1]))] for j in range(len(raw_map))]

  # Format map so that objects = 1 and free space = 0
  # Iterate through each pixel
  for i in range(len(raw_map)):
    for j in range(len(raw_map[i])):

      if (raw_map[i][j] == 0):
        # Pixel is object
        map[i][j] = 1
      else:
        # Pixel is free space
        map[i][j] = 0


  # Contains a series of points we want to navigate to. Each point will be navigated to in order.
  destinations = []

  # List containing all the paths taken
  paths = []

  # Set the goal
  if (obj[0] != -1): # We saw the object while we were mapping. Let's navigate straight to the object.
    destinations.append(obj)
  else: # We didn't see the object while we were mapping. Let's navigate from most probable room to least probable room.

    # Match the ordered dictionnary of probabilities to the dictionnary of locations and append to the goal.
    for room_probability in p_rooms:
      destinations.append((rooms[room_probability[0]][0], rooms[room_probability[0]][1]))

  # Navigate to the goal
  for destination in destinations:
    # Navigate to the destination
    paths.append(astar(map, current_pose, destination))

    # Set the current pose to the end of the path
    current_pose = paths[-1][-1]

    # Since we do not have a robot, a random function will determine if we found the object in the room.
    # If we already saw the object while we were mapping, this probability function does not apply and is always true
    found = (True if (rand.random() > 0.5) or (obj[0] != -1) else False)

    if (found):
      break
  
  # Convert map to RGB
  map_image = map_image.convert('RGB')

  # Draw the path onto the map
  for path in paths:
    for point in path:
      map_image.putpixel((point[1],point[0]), (0, 0, 255))

  return map_image

#seems like below line was added for testing and is not needed
#path_planner('/content/drive/MyDrive/Colab Notebooks/map.pgm', obj, p_rooms, rooms, current_pose)

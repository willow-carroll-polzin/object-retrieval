def path_planner(map, obj, p_room, rooms, current_pose):
  # Contains a series of points we want to navigate to. Each point will be navigated to in order.
  destinations = []

  # Set the goal
  if (obj) # We saw the object while we were mapping. Let's navigate straight to the object.
    destinations.append(obj)
  else # We didn't see the object while we were mapping. Let's navigate from most probable room to least probable room.
    
    # Convert the tuples in a dictionnary for ease and store in order of most probable room
    dict_p_rooms = sorted(dict((y, x) for x, y in p_rooms))

    # Match the ordered dictionnary of probabilities to the dictionnary of locations and append to the goal.
    for room in dict_p_rooms
      destinations.append([rooms[room][0], rooms[room][1]])

  # Navigate to the goal
  for destination in destinations
    # A star
    
    # Since we do not have a robot, a random function will determine if we found the object in the room.
    # If we already saw the object while we were mapping, this probability function does not apply and is always true
    found = (True if (rand() > 0.2) or (obj) else False)

    if (found)
      break

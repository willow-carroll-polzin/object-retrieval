def label_map(rooms, path):

  # Size of window filter
  window_size = 10

  # Counter that moves the window along the path
  path_idx = 1

  # Setup a dictionnary to store all of the rooms and their positions
  room_pos = {}

  # Initialize the dictionnary to contain all of the found rooms
  for room in rooms:
    room_pos.update({room: []})

  for pose in path:

    # Setup a dictionnary to keep track of the occurance of each room within the filter
    room_occ = {}

    # Initialize the dictionnary to contain all of the found rooms
    for room in rooms:
      room_occ.update({room: 0})

    # Running total of the coordinates in the window. Used to find the centroid coordinate of the window
    rc_x = 0
    rc_y = 0

    # Scan over the window
    for i in range(window_size):

      # Add to the running total of the coordinates in the window. Used to find the centroid coordinate of the window
      rc_x = rc_x + path[path_idx + i][0]
      rc_y = rc_y + path[path_idx + i][1]

      # Used to determine the room type with the highest occurence in the window
      max_w = 0
      in_room = ""

      # Scan over every room. Used to compare each individual pose to the known rooms
      for room in rooms:

        # Current room is the same as in the pose
        if (room == path[path_idx + i][2]):
          # Increment the counter in the dicitionnary for that specific room
          room_occ.update({room: room_occ[room]+1})
        
        # Update the room with the highest occurance
        if (room_occ[room]/window_size > max_w):
          max_w = room_occ[room]/window_size
          in_room = room
      
    # Take the average of the running total of coordinates. This is the centroid of the room
    room_cord_x = rc_x / window_size
    room_cord_y = rc_y / window_size
    
    # Move the window along
    if (path_idx < len(path)/2):
      path_idx = path_idx + 1
    else:
      break

    # Add the coordinates of each room to a running total. This is done to get the average position of each room.
    for room in rooms:
      if (in_room == room):
        room_pos[room].append([room_cord_x, room_cord_y])


  # Calculate the average room coordinate based off of the running total
  for room in rooms:
    x_tot = 0
    y_tot = 0
    room_len = len(room_pos[room])

    # Increment the running total
    for i in range(room_len):
      x_tot = x_tot + room_pos[room][i][0];
      y_tot = y_tot + room_pos[room][i][1];

    # Clear the list and replace the entry with the centroid of the room.
    room_pos[room].clear()
    room_pos[room].append([x_tot/room_len, y_tot/room_len])

  return(room_pos)

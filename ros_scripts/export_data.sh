# Save the occupancy map
rosrun map_server map_saver -f map map:=/occupancy

# Move the bag to where export.launch wants it
sudo cp ./data.bag /opt/ros/noetic/share/image_view/export.bag

# Extract the RGB data and save it to ~/.ros/
roslaunch export.launch

# Export the pose data to csv
rostopic echo -b data.bag -p /tf > pose.csv

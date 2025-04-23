FROM osrf/ros:noetic-desktop-full

# Make sure ROS environment is sourced on container start
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Default command to run roscore
CMD ["roscore"]

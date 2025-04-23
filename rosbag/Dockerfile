FROM osrf/ros:noetic-desktop-full

RUN mkdir -p rosbag

COPY lidar2.bag /rosbag/my_data.bag

ENV ROSBAG_FILE=/rosbag/my_data.bag

RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

CMD bash -c "source /opt/ros/noetic/setup.bash && rosbag play --loop $ROSBAG_FILE"
xhost local:root

XAUTH=/tmp/.docker.xauth

sudo docker run -it \
    --name=noetic \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --env="XAUTHORITY=$XAUTH" \
    --volume="$XAUTH:$XAUTH" \
    --net=host \
    --privileged \
    osrf/ros:noetic-desktop-full \
    bash

echo "Done."

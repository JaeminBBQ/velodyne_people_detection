1. Clone Repository
2. Install Docker (https://docs.docker.com/engine/install/ubuntu/)
3. Navigate to rosbag directory. Have desired bag file and adjust dockerfile accordingly and run sudo docker build . -t bag
4. Navigate to roscore directory and run sudo docker build . -t core
5. Navigate to OpenPCDet directory and run sudo docker build . -t openpcdet-trained
6. Once execed in, go to tools directory, run python3 train.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar.yaml
7. Copy poopoo.py into container sudo docker cp /home/jaeminbbq/Projects/rosenv/OpenPCDet/poopoo.py noetic_gpu:/OpenPCDet/tools
docker run --net=host -d <name>

docker build -t <name> .

--gpus all
--runtime=nvidia
-it

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuration

python3.9 
pip install "numpy<2.0"

d run -it --net=host --gpus all temp

WORKDIR /OpenPCDet/tools
RUN python3 train.py --cfg_file /OpenPCDet/tools/cfgs/custom_models/pointpillar.yaml

sudo docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

Need nvidia toolkit

docker commit <container_id_or_name> openpcdet-trained-image

python3 tools/demo.py --cfg_file /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar.yaml \
    --ckpt /OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth \
    --data_path /OpenPCDet/data/custom/points/

export PYTHONPATH=$PYTHONPATH:/OpenPCDet 

docker cp /home/jaeminbbq/Projects/rosenv/OpenPCDet/poopoo.py noetic_gpu:/OpenPCDet/tools
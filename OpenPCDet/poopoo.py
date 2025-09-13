import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import tf.transformations
import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
import argparse
import glob
from pathlib import Path

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        self.ext = '.npy'
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            print("[DEBUG] - RUNNING .npy")
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

class Detector:
    def __init__(self):
        rospy.init_node('poopoo')

        self.marker_pub = rospy.Publisher('/poo_poo', Marker, queue_size=10)

        cfg_file = "/OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/pointpillar.yaml"
        ckpt = "/OpenPCDet/output/OpenPCDet/tools/cfgs/custom_models/pointpillar/default/ckpt/checkpoint_epoch_80.pth"
        data_path = "/OpenPCDet/data/custom/points/000000.npy"
        ext = ".npy"
        logger = common_utils.create_logger()
        cfg_from_yaml_file(cfg_file, cfg)

        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=Path(data_path), ext=ext, logger=logger
        )

        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=ckpt, logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

        rospy.Subscriber('/velodyne_points', PointCloud2, self.pointcloud_callback)

    def publish_box(self, box, score, label, marker_id=0):
        x, y, z, dx, dy, dz, yaw = box
        marker = Marker()
        marker.header.frame_id = "velodyne"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "detections"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        quat = tf.transformations.quaternion_from_euler(0, 0, yaw)
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        marker.scale.x = dx
        marker.scale.y = dy
        marker.scale.z = dz

        marker.color = ColorRGBA(1.0, 0.0, 0.0, 0.5)
        marker.lifetime = rospy.Duration(0.2)

        self.marker_pub.publish(marker)
    
    def pointcloud_callback(self, msg):
        cloud_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        cloud_npy = np.array(cloud_points, dtype=np.float32)

        input_dict = {
            'points': cloud_npy,
            'frame_id': 0,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        with torch.no_grad():
            data_dict = self.demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
            
            for i in range(len(pred_dicts[0]['pred_boxes'])):
                box = pred_dicts[0]['pred_boxes'][i].cpu().numpy()
                score = pred_dicts[0]['pred_scores'][i].item()
                label = pred_dicts[0]['pred_labels'][i].item()

                if score > 0.3:
                    self.publish_box(box, score, label, marker_id=i)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = Detector()
        detector.run()
    except rospy.ROSInterruptException:
        pass

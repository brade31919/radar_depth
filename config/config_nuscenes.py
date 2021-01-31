"""
This file contains configurations of the NuScenes dataset
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from attrdict import AttrDict
import os


# Define the configurations for kitti dataset
class config_nuscenes(object):
    PROJECT_ROOT = "YOUR_PATH/radar_depth"
    dataset_mode = "full"

    # Data path configuration
    DATASET_ROOT = "DATASET_PATH"

    # Some parameters
    TRAIN_VAL_SEED = 100
    VAL_RATIO = 0.1

    # Define the orientation mode
    # ver1: only front and back
    # ver2: all directions
    version = "ver3"
    lidar_sweeps = 1
    radar_sweeps = 1

    scaling = True
    scale_factor = 0.5

    # [DORN] transform parameters
    DORN_transform_config = AttrDict({
        "crop_size_train": [385, 513],
        "rotation_factor": 5.,
        "scale_factor_train": [1., 1.5],
        "crop_size_val": [385, 513],
        "scale_factor_val": 1.
    })
    # [sparse-to-dense] transform parameters
    sparse_transform_config = AttrDict({
        "crop_size_train": [450, 800],
        "rotation_factor": 5.,
        "scale_factor_train": [1., 1.5],
        "crop_size_val": [450, 800],
        "eval_size": [450, 800],
        "scale_factor_val": 1.
    })

    # Dataset export path
    EXPORT_ROOT = os.path.join(DATASET_ROOT, "Nuscenes_depth")

    # Always do ver2
    if version in ["ver1", "ver2"]:
        export_name = "ver2_lidar%d_radar%d" % (lidar_sweeps, radar_sweeps)
        radar_export_name = None
    elif version == "ver3":
        export_name = "ver2_lidar%d_radar%d" % (lidar_sweeps, radar_sweeps)
        radar_export_name = "ver2_lidar1_radar3_radar_only"
    else:
        raise ValueError("Unknow dataset version. we currently only support ver1~ver3.")
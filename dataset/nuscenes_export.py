"""
Export the datapoint to disk
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Add system path for fast debugging
import sys
sys.path.append("../")

import numpy as np
from dataset.nuscenes_dataset import Nuscenes_dataset
from config.config_nuscenes import config_nuscenes as cfg
from dataset import transforms as transforms
import h5py
import os
from tqdm import tqdm
import ipdb


def create_h5_file(output_filename, data, save_keys=None):
    # Compress depth maps
    if "lidar_depth" in save_keys:
        data["lidar_depth"] = (data["lidar_depth"] * 256).astype(np.int16)
    if "radar_depth" in save_keys:
        data["radar_depth"] = (data["radar_depth"] * 256).astype(np.int16)

    # Create file objects
    with h5py.File(output_filename, "w") as f:
        # Iterate through all the key value pairs in the object
        for key, value in data.items():
            if key in save_keys:
                f.create_dataset(name=key,
                                 shape=value.shape,
                                 dtype=value.dtype,
                                 data=value)


def parse_h5_file(file_path):
    # Check if file exists
    if not os.path.exists(file_path):
        raise ValueError("[Error] File does not exist.")

    # Read file
    output_dict = {}
    with h5py.File(file_path, "r") as f:
        for key_name in f:
            output_dict[key_name] = np.array(f[key_name])

    # Decompress depth
    output_dict["lidar_depth"] = output_dict["lidar_depth"] / 256.
    output_dict["radar_depth"] = output_dict["radar_depth"] / 256.

    return output_dict


# Export dataset according to mode
def export_dataset(datapoints, dataset, mode="train"):
    # Define the export path
    # export_path = os.path.join(cfg.EXPORT_ROOT, cfg.export_name, mode)
    export_path = os.path.join(cfg.EXPORT_ROOT, cfg.export_name + "_radar_only", mode)

    ###########################
    ## Add radar only option ##
    ###########################
    # export_path = export_path + "_radar_only"
    save_keys = ['radar_depth', 'radar_depth_points', 'radar_points', 'radar_raw_points']
    ###########################

    if not os.path.exists(export_path):
        os.makedirs(export_path)

    # Iterate through all the datapoints
    for i in tqdm(range(len(datapoints)), ascii=True):
        datapoint = datapoints[i]

        # Get the index and orientation
        index = datapoint[0]
        orientation = datapoint[1]

        # Get output filename
        output_filename = "%07d_%s.h5" % (index, orientation)
        output_filename = os.path.join(export_path, output_filename)

        # Get data
        data = dataset.get_data(datapoint, mode=mode)

        # ipdb.set_trace()

        # Export the data
        create_h5_file(output_filename, data, save_keys=save_keys)

        # print("[ %d / %d ]"%(i, len(datapoints)))


if __name__ == "__main__":
    # Initialize dataset object
    dataset = Nuscenes_dataset(mode=cfg.dataset_mode)

    # Get all train / val samples
    train_data_points = dataset.get_datapoints("train")
    val_data_points = dataset.get_datapoints("val")

    # Export training set
    export_dataset(train_data_points, dataset, "train")

    # Export testing set
    export_dataset(val_data_points, dataset, "val")
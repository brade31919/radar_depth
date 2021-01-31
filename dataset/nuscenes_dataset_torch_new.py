"""
This file implement the dataset object for pytorch
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Add system path for fast debugging
import sys
sys.path.insert(0, "/cluster/home/julin/workspace/Semester_project_release")

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from dataset.nuscenes_dataset import Nuscenes_dataset
from config.config_nuscenes import config_nuscenes as cfg
from dataset.dense_to_sparse import UniformSampling, LidarRadarSampling
from dataset import transforms as transforms
from dataset.radar_preprocessing import filter_radar_points_gt
import math
import h5py
import pickle
import matplotlib.pyplot as plt
to_tensor = transforms.ToTensor()


####################################
## Sparsifier Documentations:
## 1. uniform: Uniformly sampled LiDAR points.
## 2. lidar_radar: Sampled LiDAR points using the radar pattern.
## 3. radar: raw radar points (accumulated from three time steps.
## 4. radar_filtered: Filtered radar points using the heuristic algorithm.
## 5. radar_filtered2: Filtered radar points using the trained point classifier.
####################################

# Define the dataset object for torch
class nuscenes_dataset_torch(Dataset):
    def __init__(self, 
            mode="train", transform_mode="DORN", modality="rgb",
            sparsifier=None, num_samples=0, max_depth=100.,
        ):
        super(nuscenes_dataset_torch, self).__init__()
        print("[Info] Initializing exported nuscenes dataset")
        self.mode = mode
        self.filename_dataset = self.get_filename_dataset()
        print("\t Mode: ", self.mode)
        print("\t Version: ", cfg.version)
        print("\t Data counts: ", self.filename_dataset["length"])

        # Check modalities
        self.avail_modality = ["rgb", "rgbd"]
        if not modality in self.avail_modality:
            raise ValueError("[Error] Unsupported modality. Consider ", self.avail_modality)

        self.modality = modality
        print("\t Modality: ", self.modality)

        # Check sparsifier and modality
        if (self.modality == "rgb"):
            self.sparsifier = "radar"
            self.num_samples = num_samples
            self.max_depth = max_depth

        elif (self.modality == "rgbd") and (sparsifier is None):
            # If rgbd and not sparsifier, then use radar
            self.sparsifier = "radar"
            self.num_samples = num_samples
            self.max_depth = max_depth

        elif (self.modality == "rgbd") and (sparsifier is not None):
            # If sparsifier is provided then check if it's valid
            if not sparsifier in ["uniform", "lidar_radar", "radar",
                                  "radar_filtered", "radar_filtered2"]:
                raise ValueError("[Error] Invalid sparsifier.")

            assert num_samples is not None
            self.sparsifier = sparsifier
            self.num_samples = num_samples
            self.max_depth = max_depth
            # Initialize uniform sampler
            if self.sparsifier == "uniform":
                self.sparsifier_func = UniformSampling(num_samples, max_depth)
            # Initialize lidar_radar sampler
            elif self.sparsifier == "lidar_radar":
                self.sparsifier_func = LidarRadarSampling(num_samples, max_depth)
            # Radar will be handled in the end of transform, no sparsifier_func is required
            elif (self.sparsifier == "radar") or \
                 (self.sparsifier == "radar_filtered") or \
                 (self.sparsifier == "radar_filtered2"):
                pass
            else:
                raise NotImplementedError

        print("\t Sparsifier: ", self.sparsifier)

        # Further get the day-night split table.
        day_night_info = self.get_day_night_info()
        self.train_daynight_table = day_night_info["train"]
        self.train_daynight_count = day_night_info["train_count"]
        self.test_daynight_table = day_night_info["test"]
        self.test_daynight_count = day_night_info["test_count"]
        print("\t Day-Night info:")
        print("\t\t Train:")
        print("\t\t\t Day: %d" %(self.train_daynight_count["day"]))
        print("\t\t\t Day + Rain: %d" %(self.train_daynight_count["day_rain"]))
        print("\t\t\t Night: %d" %(self.train_daynight_count["night"]))
        print("\t\t\t Night + Rain: %d" %(self.train_daynight_count["night_rain"]))
        print("\t\t Test:")
        print("\t\t\t Day: %d" %(self.test_daynight_count["day"]))
        print("\t\t\t Day + Rain: %d" %(self.test_daynight_count["day_rain"]))
        print("\t\t\t Night: %d" %(self.test_daynight_count["night"]))
        print("\t\t\t Night + Rain: %d" %(self.test_daynight_count["night_rain"]))
        print("-----------------------------------------")

        # Check transform mode
        assert transform_mode in ["DORN", "sparse-to-dense"]
        self.transform_mode = transform_mode
        if self.transform_mode == "DORN":
            self.t_cfg = cfg.DORN_transform_config
        elif self.transform_mode == "sparse-to-dense":
            self.t_cfg = cfg.sparse_transform_config

        # Define outut size
        if self.mode == "train":
            self.output_size = self.t_cfg.crop_size_train
        else:
            self.output_size = self.t_cfg.crop_size_val

    # Create filename dataset from the exported nuscenes dataset
    def get_filename_dataset(self):
        dataset_root = os.path.join(cfg.EXPORT_ROOT, cfg.export_name)

        # Use different root for different mode
        if self.mode == "train":
            dataset_root = os.path.join(dataset_root, "train")
        elif self.mode == "val":
            dataset_root = os.path.join(dataset_root, "val")
        else:
            raise ValueError("[Error] Unknown dataset mode")

        # Add different radar root for version 3
        if cfg.version == "ver3":
            assert cfg.radar_export_name is not None
            dataset_root_radar = os.path.join(cfg.EXPORT_ROOT, cfg.radar_export_name)
            if self.mode == "train":
                dataset_root_radar = os.path.join(dataset_root_radar, "train")
            elif self.mode == "val":
                dataset_root_radar = os.path.join(dataset_root_radar, "val")

        # Get all filenames in the dataroot
        filenames = os.listdir(dataset_root)
        filenames = [_ for _ in filenames if _.endswith(".h5")]

        # Get subset of filenames given dataset version
        if cfg.version in ["ver1", "ver3"]:
            ver1_ori = ["front", "back"]
            filenames = [_ for _ in filenames if os.path.splitext(_)[0].split("_")[-1] in ver1_ori]

        assert len(filenames) > 0
        # Add to full data path
        filenames_original = [os.path.join(dataset_root, _) for _ in filenames]

        # Add special case for version3
        if cfg.version == "ver3":
            filenames_radar = [os.path.join(dataset_root_radar, _) for _ in filenames]
            return {
                "datapoints": filenames_original,
                "datapoints_radar": filenames_radar,
                "length": len(filenames)
            }

        else:
            return {
                "datapoints": filenames_original,
                "length": len(filenames)
            }

    # Get h5 data given data path
    def get_data(self, datapoint):
        # Check if file exists
        if not os.path.exists(datapoint):
            raise ValueError("[Error] File does not exist.")

        # Read file
        output_dict = {}
        with h5py.File(datapoint, "r") as f:
            for key_name in f:
                output_dict[key_name] = np.array(f[key_name])

        # Decompress depth
        if "lidar_depth" in output_dict.keys():
            output_dict["lidar_depth"] = output_dict["lidar_depth"] / 256.
        if "radar_depth" in output_dict.keys():
            output_dict["radar_depth"] = output_dict["radar_depth"] / 256.

        return output_dict

    # Sampling sparse depth points from lidar
    def get_sparse_depth(self, input_depth, radar_depth=None):
        # Check if the sparsifier is valid
        if not self.sparsifier in ["uniform", "lidar_radar"]:
            raise ValueError("[Error] Invalid lidar sparsifier.")

        if self.sparsifier == "uniform":
            mask_keep = self.sparsifier_func.dense_to_sparse(input_depth)

        elif self.sparsifier == "lidar_radar":
            assert radar_depth is not None
            mask_keep = self.sparsifier_func.dense_to_sparse(input_depth, radar_depth)
            mask_keep = torch.tensor(mask_keep[..., None].transpose(2, 0, 1)).to(torch.bool)

        # ipdb.set_trace()
        sparse_depth = torch.zeros(input_depth.shape)
        sparse_depth[mask_keep] = input_depth[mask_keep]
        return sparse_depth

    # Get the exported day night info
    def get_day_night_info(self):
        # Check the default
        file_path = os.path.join(cfg.EXPORT_ROOT, "nuscenes_day_night_info.pkl")

        if not os.path.exists(file_path):
            raise ValueError("[Error] Can't find the day-night info pickle file in %s" % (file_path))
        
        # Load the file
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        return data

    # Return the length of a dataset
    def __len__(self):
        return self.filename_dataset["length"]

    # Define the transform for train
    def transform_train(self, input_data):
        # import ipdb; ipdb.set_trace()
        # Fetch the data
        rgb = np.array(input_data["image"]).astype(np.float32)
        lidar_depth = np.array(input_data["lidar_depth"]).astype(np.float32)
        radar_depth = np.array(input_data["radar_depth"]).astype(np.float32)
        if 'index_map' in input_data.keys():
            index_map = np.array(input_data["index_map"]).astype(np.int)

        # Define augmentation factor
        scale_factor = np.random.uniform(self.t_cfg.scale_factor_train[0], self.t_cfg.scale_factor_train[1])  # random scaling
        angle_factor = np.random.uniform(-self.t_cfg.rotation_factor, self.t_cfg.rotation_factor)  # random rotation degrees
        flip_factor = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # Compose customized transform for RGB and Depth separately
        color_jitter = transforms.ColorJitter(0.2, 0.2, 0.2)
        resize_image = transforms.Resize(scale_factor, interpolation="bilinear")
        resize_depth = transforms.Resize(scale_factor, interpolation="nearest")

        # # First, we uniformly downsample all the images by half
        # resize_image_initial = transforms.Resize(0.5, interpolation="bilinear")
        # resize_depth_initial = transforms.Resize(0.5, interpolation="nearest")

        # Then, we add model-aware resizing
        if self.transform_mode == "DORN":
            if cfg.scaling is True:
                h, w, _ = tuple((np.array(rgb.shape)).astype(np.int32))
            else:
                h, w, _ = tuple((np.array(rgb.shape)* 0.5).astype(np.int32))

            # ipdb.set_trace()
            h_new = self.t_cfg.crop_size_train[0]
            w_new = w
            resize_image_method = transforms.Resize([h_new, w_new], interpolation="bilinear")
            resize_depth_method = transforms.Resize([h_new, w_new], interpolation="nearest")
        elif self.transform_mode == "sparse-to-dense":
            h_new = self.t_cfg.crop_size_train[0]
            w_new = self.t_cfg.crop_size_train[1]
            resize_image_method = transforms.Resize([h_new, w_new], interpolation="bilinear")
            resize_depth_method = transforms.Resize([h_new, w_new], interpolation="nearest")

        # Get the border of random crop
        h_scaled, w_scaled = math.floor(h_new * scale_factor), math.floor((w_new * scale_factor))
        h_bound, w_bound = h_scaled - self.t_cfg.crop_size_train[0], w_scaled - self.t_cfg.crop_size_train[1]
        h_startpoint = round(np.random.uniform(0, h_bound))
        w_startpoint = round(np.random.uniform(0, w_bound))

        # Compose the transforms for RGB
        transform_rgb = transforms.Compose([
            transforms.Rotate(angle_factor),
            resize_image,
            transforms.Crop(h_startpoint, w_startpoint, self.t_cfg.crop_size_train[0], self.t_cfg.crop_size_train[1]),
            transforms.HorizontalFlip(flip_factor)
        ])

        # Compose the transforms for Depth
        transform_depth = transforms.Compose([
            transforms.Rotate(angle_factor),
            resize_depth,
            transforms.Crop(h_startpoint, w_startpoint, self.t_cfg.crop_size_train[0], self.t_cfg.crop_size_train[1]),
            transforms.HorizontalFlip(flip_factor)
        ])

        # Perform transform on rgb data
        # ToDo: whether we need to - imagenet mean here
        rgb = transform_rgb(rgb)
        rgb = color_jitter(rgb)
        rgb = rgb / 255.

        # Perform transform on lidar depth data
        lidar_depth /= float(scale_factor)
        lidar_depth = transform_depth(lidar_depth)

        rgb = np.array(rgb).astype(np.float32)
        lidar_depth = np.array(lidar_depth).astype(np.float32)

        rgb = to_tensor(rgb)
        lidar_depth = to_tensor(lidar_depth)

        # Perform transform on radar depth data
        radar_depth /= float(scale_factor)
        radar_depth = transform_depth(radar_depth)

        radar_depth = np.array(radar_depth).astype(np.float32)
        radar_depth = to_tensor(radar_depth)

        # Perform transform on index map
        if 'index_map' in input_data.keys():
            index_map = transform_depth(index_map)
            index_map = np.array(index_map).astype(np.int)
            index_map = to_tensor(index_map)
            index_map = index_map.unsqueeze(0)

        # Normalize rgb using imagenet mean and std
        # ToDo: only do imagenet normalization on DORN
        if self.transform_mode == "DORN":
            rgb = transforms.normalization_imagenet(rgb)

        if self.sparsifier == "radar_filtered":
            ####################
            ## Filtering part ##
            ####################
            # Indicating the invalid entries
            invalid_mask = ~ input_data['valid_mask']
            invalid_index = np.where(invalid_mask)[0]
            invalid_index_mask = invalid_index[None, None, ...].transpose(2, 0, 1)

            # Constructing mask for dense depth
            dense_mask = torch.ByteTensor(np.sum(index_map.numpy() == invalid_index_mask, axis=0))
            radar_depth_filtered = radar_depth.clone()
            radar_depth_filtered[dense_mask.to(torch.bool)] = 0.
            radar_depth_filtered = radar_depth_filtered.unsqueeze(0)

        if self.sparsifier == "radar_filtered2":
            ######################################
            ## Filtering using predicted labels ##
            ######################################
            invalid_mask = ~ input_data['pred_labels']
            invalid_index = np.where(invalid_mask)[0]
            invalid_index_mask = invalid_index[None, None, ...].transpose(2, 0, 1)

            dense_mask = torch.ByteTensor(np.sum(index_map.numpy() == invalid_index_mask, axis=0))
            radar_depth_filtered2 = radar_depth.clone()
            radar_depth_filtered2[dense_mask.to(torch.bool)] = 0.
            radar_depth_filtered2 = radar_depth_filtered2.unsqueeze(0)
            ######################################

        lidar_depth = lidar_depth.unsqueeze(0)
        radar_depth = radar_depth.unsqueeze(0)

        # Return different data for different modality
        if self.modality == "rgb":
            inputs = rgb
        elif self.modality == "rgbd":
            if self.sparsifier == "radar":
                # Filter out the the points exceeding max_depth
                mask = (radar_depth > self.max_depth)
                radar_depth[mask] = 0
                inputs = torch.cat((rgb, radar_depth), dim=0)
            # Using the generated groundtruth
            elif self.sparsifier == "radar_filtered":
                # Filter out the points exceeding max_depth
                mask = (radar_depth_filtered > self.max_depth)
                radar_depth_filtered[mask] = 0
                inputs = torch.cat((rgb, radar_depth_filtered), dim=0)
            # Using the learned classifyer
            elif self.sparsifier == "radar_filtered2":
                # Filter out the points exceeding max_depth
                mask = (radar_depth_filtered2 > self.max_depth)
                radar_depth_filtered2[mask] = 0
                inputs = torch.cat((rgb, radar_depth_filtered2), dim=0)
            else:
                s_depth = self.get_sparse_depth(lidar_depth, radar_depth)
                inputs = torch.cat((rgb, s_depth), dim=0)
        else:
            raise ValueError("[Error] Unsupported modality. Consider ", self.avail_modality)
        labels = lidar_depth

        # Gathering output results
        output_dict = {
                "rgb": rgb,
                "lidar_depth": lidar_depth,
                "radar_depth": radar_depth,
                "inputs": inputs,
                "labels": labels
            }
        if self.sparsifier == "radar_filtered":
            output_dict["radar_depth_filtered"] = radar_depth_filtered

        if self.sparsifier == "radar_filtered2":
            output_dict["radar_depth_filtered2"] = radar_depth_filtered2

        if 'index_map' in input_data.keys():
            output_dict["index_map"] = index_map

        return output_dict

    # Define the transform for val
    def transform_val(self, input_data):
        rgb = np.array(input_data["image"]).astype(np.float32)
        lidar_depth = np.array(input_data["lidar_depth"]).astype(np.float32)
        radar_depth = np.array(input_data["radar_depth"]).astype(np.float32)
        if 'index_map' in input_data.keys():
            index_map = np.array(input_data["index_map"]).astype(np.int)

        # Then, we add model-aware resizing
        if self.transform_mode == "DORN":
            if cfg.scaling is True:
                h, w, _ = tuple((np.array(rgb.shape)).astype(np.int32))
            else:
                h, w, _ = tuple((np.array(rgb.shape) * 0.5).astype(np.int32))

            h_new = self.t_cfg.crop_size_train[0]
            w_new = w
            resize_image_method = transforms.Resize([h_new, w_new], interpolation="bilinear")
            resize_depth_method = transforms.Resize([h_new, w_new], interpolation="nearest")
        elif self.transform_mode == "sparse-to-dense":
            h_new = self.t_cfg.crop_size_train[0]
            w_new = self.t_cfg.crop_size_train[1]
            resize_image_method = transforms.Resize([h_new, w_new], interpolation="bilinear")
            resize_depth_method = transforms.Resize([h_new, w_new], interpolation="nearest")

        transform_rgb = transforms.Compose([
            # resize_image_method,
            transforms.CenterCrop(self.t_cfg.crop_size_val)
        ])
        transform_depth = transforms.Compose([
            # resize_depth_method,
            transforms.CenterCrop(self.t_cfg.crop_size_val)
        ])

        rgb = transform_rgb(rgb)
        rgb = rgb / 255.
        lidar_depth = transform_depth(lidar_depth)

        rgb = np.array(rgb).astype(np.float32)
        lidar_depth = np.array(lidar_depth).astype(np.float32)

        rgb = to_tensor(rgb)
        lidar_depth = to_tensor(lidar_depth)

        radar_depth = transform_depth(radar_depth)
        radar_depth = np.array(radar_depth).astype(np.float32)
        radar_depth = to_tensor(radar_depth)

        # Perform transform on index map
        if 'index_map' in input_data.keys():
            index_map = transform_depth(index_map)
            index_map = np.array(index_map).astype(np.int)
            index_map = to_tensor(index_map)
            index_map = index_map.unsqueeze(0)

        # Normalize to imagenet mean and std
        if self.transform_mode == "DORN":
            rgb = transforms.normalization_imagenet(rgb)

        ####################
        ## Filtering part ##
        ####################
        if self.sparsifier == "radar_filtered":
            # Indicating the invalid entries
            invalid_mask = ~ input_data['valid_mask']
            invalid_index = np.where(invalid_mask)[0]
            invalid_index_mask = invalid_index[None, None, ...].transpose(2, 0, 1)

            # Constructing mask for dense depth
            dense_mask = torch.ByteTensor(np.sum(index_map.numpy() == invalid_index_mask, axis=0))
            radar_depth_filtered = radar_depth.clone()
            radar_depth_filtered[dense_mask.to(torch.bool)] = 0.
            radar_depth_filtered = radar_depth_filtered.unsqueeze(0)
            # ipdb.set_trace()
            ####################

        ######################################
        ## Filtering using predicted labels ##
        ######################################
        if self.sparsifier == "radar_filtered2":
            # ipdb.set_trace()
            invalid_mask = ~ input_data['pred_labels']
            invalid_index = np.where(invalid_mask)[0]
            invalid_index_mask = invalid_index[None, None, ...].transpose(2, 0, 1)

            dense_mask = torch.ByteTensor(np.sum(index_map.numpy() == invalid_index_mask, axis=0))
            radar_depth_filtered2 = radar_depth.clone()
            radar_depth_filtered2[dense_mask.to(torch.bool)] = 0.
            radar_depth_filtered2 = radar_depth_filtered2.unsqueeze(0)
            ######################################

        lidar_depth = lidar_depth.unsqueeze(0)
        radar_depth = radar_depth.unsqueeze(0)

        # Return different data for different modality
        ################ Input sparsifier #########
        if self.modality == "rgb":
            inputs = rgb
        elif self.modality == "rgbd":
            if self.sparsifier == "radar":
                # Filter out the the points exceeding max_depth
                mask = (radar_depth > self.max_depth)
                radar_depth[mask] = 0
                inputs = torch.cat((rgb, radar_depth), dim=0)
            elif self.sparsifier == "radar_filtered":
                # Filter out the points exceeding max_depth
                mask = (radar_depth_filtered > self.max_depth)
                radar_depth_filtered[mask] = 0
                inputs = torch.cat((rgb, radar_depth_filtered), dim=0)
            # Using the learned classifyer
            elif self.sparsifier == "radar_filtered2":
                # Filter out the points exceeding max_depth
                mask = (radar_depth_filtered2 > self.max_depth)
                radar_depth_filtered2[mask] = 0
                inputs = torch.cat((rgb, radar_depth_filtered2), dim=0)
            else:
                s_depth = self.get_sparse_depth(lidar_depth, radar_depth)
                inputs = torch.cat((rgb, s_depth), dim=0)
        else:
            raise ValueError("[Error] Unsupported modality. Consider ", self.avail_modality)
        labels = lidar_depth

        output_dict = {
                "rgb": rgb,
                "lidar_depth": lidar_depth,
                "radar_depth": radar_depth,
                "inputs": inputs,
                "labels": labels
            }

        if self.sparsifier == "radar_filtered":
            output_dict["radar_depth_filtered"] = radar_depth_filtered

        if self.sparsifier == "radar_filtered2":
            output_dict["radar_depth_filtered2"] = radar_depth_filtered2

        # For 'index_map' compatibility
        if 'index_map' in input_data.keys():
            output_dict["index_map"] = index_map

        return output_dict

    # Add index map and perform valid check
    def filter_radar_points(self, input_data):
        # Fetch data
        radar_points = input_data['radar_points']
        radar_depth_points = input_data['radar_depth_points']

        # Construct index map
        depth_loc = (radar_points[:2, :].T).astype(np.int32)
        point_index = np.arange(0, radar_points.shape[1], 1)
        index_map = - np.ones(input_data['image'].shape[:2])
        index_map[depth_loc[:, 1], depth_loc[:, 0]] = point_index

        input_data['index_map'] = index_map

        # Filter the radar points
        filtered_data = filter_radar_points_gt(input_data['radar_points'],
                                            input_data['radar_depth_points'],
                                            input_data['lidar_points'],
                                            input_data['lidar_depth_points'])

        # Record the masked depth points
        input_data['radar_points_filtered'] = filtered_data['radar_points']
        input_data['radar_depth_points_filtered'] = filtered_data['radar_depth']
        input_data['valid_mask'] = filtered_data['valid_mask']

        if self.sparsifier == "radar_filtered2":
            raise NotImplementedError("[Error] The filtering method using point classifier is not supported in the released code.")

        return input_data

    # Perform transform on radar pointclouds
    def transform_point(self, point_data):
        points = point_data["radar_points_raw"]
        labels = point_data["radar_points_label"][..., None]

        # shuffle the points
        if self.mode == "train":
            tmp = np.concatenate((points, labels), axis=-1)
            np.random.shuffle(tmp)
            points = tmp[:, :-1]
            labels = tmp[:, -1][..., None]

        # Pad the points to 512
        num_points = points.shape[0]
        target_size = 512
        output_points = np.repeat(points[-1, :][None, ...], target_size, axis=0)
        output_points[:num_points, :] = points
        output_labels = np.repeat(labels[-1, :][None, ...], target_size, axis=0)
        output_labels[:num_points, :] = labels

        # Create valid mask
        mask = np.zeros([target_size, 1])
        mask[:num_points] = 1

        return {
            "radar_points_raw": to_tensor(output_points),
            "radar_points_label": to_tensor(output_labels),
            "radar_points_mask": to_tensor(mask)
        }

    # Define the getitem method
    def __getitem__(self, index):
        # Get data from given index
        datapoint = self.filename_dataset["datapoints"][index]
        data = self.get_data(datapoint)

        # Get the daynight info
        daynight_key = os.path.basename(datapoint)
        if self.mode == "train":
            daynight_info = self.train_daynight_table[daynight_key]
        else:
            daynight_info = self.test_daynight_table[daynight_key]

        # Further get radar datapoints
        if cfg.version == "ver3":
            datapoint_radar = self.filename_dataset["datapoints_radar"][index]
            data_radar = self.get_data(datapoint_radar)

            for key in data_radar.keys():
                data[key] = data_radar[key]

        # Filter radar points here
        data = self.filter_radar_points(data)
        
        # Apply transforms given mode
        if self.mode == "train":
            outputs = self.transform_train(data)

        else:
            outputs = self.transform_val(data)

        # Add daynight info
        outputs["daynight_info"] = daynight_info
        
        return outputs


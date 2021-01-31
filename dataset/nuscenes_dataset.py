"""
This file warp around the nuscenes dataset => We only focus on what we want
"""
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# insert system path for fast debugging
import sys
sys.path.insert(0, "../")

from nuscenes.nuscenes import NuScenes
from nuscenes.nuscenes import NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, PointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from config.config_nuscenes import config_nuscenes as cfg
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
from pyquaternion import Quaternion
from PIL import Image
from functools import reduce
import math
import os
import pickle
import numpy as np
from config.config_nuscenes import config_nuscenes as cfg
from misc.devkit.python.read_depth import depth_read


# Define the kitti dataset abstraction object
class Nuscenes_dataset(object):
    # Initialize the dataset
    def __init__(self, mode="mini") -> None:
        # Check mode
        if not mode in ["mini", "full"]:
            raise ValueError("[Error] Unknow nuscene dataset mode. Consider using 'mini' or 'full'")
        self.mode = mode

        # Initialize nuscenes dataset API
        print("[Info] Initializing Nuscenes official database...")
        if self.mode == "mini":
            dataroot = os.path.join(cfg.DATASET_ROOT, "v1.0-mini")
            self.dataset = NuScenes(version="v1.0-mini", dataroot=dataroot)
        elif self.mode == "full":
            # dataroot = os.path.join(cfg.DATASET_ROOT, "v1.0-trainval")
            dataroot = cfg.DATASET_ROOT
            self.dataset = NuScenes(version="v1.0-trainval", dataroot=dataroot)
        self.explorer = NuScenesExplorer(self.dataset)
        print("[Info] Finished initializing Nuscenes official database!")
        # ipdb.set_trace()
        # Initialize some tables
        self.samples = self.dataset.sample
        self.scenes = self.dataset.scene
        self.num_scenes = len(self.scenes)
        self.num_samples = len(self.scenes)
        self.train_val_table = self.get_train_val_table()

        # Train related attributes
        self.train_samples = self.train_val_table["train_samples"]
        self.train_sample_tokens = self.train_val_table["train_sample_tokens"]

        # Val related attributes
        self.val_samples = self.train_val_table["val_samples"]
        self.val_sample_tokens = self.train_val_table["val_sample_tokens"]

        # Define camera keywords
        self.camera_orientations = ["front", "front_right", "front_left", "back_right", "back_left", "back"]
        self.orientation_to_camera = {
            "front": "CAM_FRONT",
            "front_right": "CAM_FRONT_RIGHT",
            "front_left": "CAM_FRONT_LEFT",
            "back_right": "CAM_BACK_RIGHT",
            "back_left": "CAM_BACK_LEFT",
            "back": "CAM_BACK"
        }

        # Define radar keywords
        self.radar_orientations = ['front', 'front_right', 'front_left', 'back_right', 'back_left']
        self.radar_keywords = ['RADAR_FRONT', 'RADAR_FRONT_RIGHT', 'RADAR_FRONT_LEFT',
                               'RADAR_BACK_RIGHT', 'RADAR_BACK_LEFT']
        self.orientation_to_radar = {
            "front": 'RADAR_FRONT',
            "front_right": 'RADAR_FRONT_RIGHT',
            "front_left": 'RADAR_FRONT_LEFT',
            "back_right": 'RADAR_BACK_RIGHT',
            "back_left": 'RADAR_BACK_LEFT'
        }

        # Now based on dataset version, we include different orientations
        # print(cfg.version)
        assert cfg.version in ["ver1", "ver2", "ver3"]
        self.version = cfg.version
        print("===============================")
        print("[Info] Use dataset %s" % (self.version))
        self.train_datapoints = self.get_datapoints("train")
        self.val_datapoints = self.get_datapoints("val")
        print("\t train datapoints: %d" % (len(self.train_datapoints)))
        print("\t val datapoints: %d" % (len(self.val_datapoints)))
        print("===============================")

        # Initialize how many sweeps to use
        self.lidar_sweeps = cfg.lidar_sweeps
        self.radar_sweeps = cfg.radar_sweeps

    ####################################
    ## Some dataset manipulation APIs ##
    ####################################
    # Easier call for get
    def get(self, table_name, token):
        return self.dataset.get(table_name, token)

    # Easier call for get_sample_data
    def get_sample_data(self, sample_data_token):
        return self.dataset.get_sample_data(sample_data_token)

    # Generate the train / val table based on the random seed
    def get_train_val_table(self):
        seed = cfg.TRAIN_VAL_SEED
        val_num = int(cfg.VAL_RATIO * self.num_scenes)

        # Get train / val scenes
        all = set(list(range(self.num_scenes)))
        np.random.seed(seed)
        val = set(np.random.choice(np.arange(0, self.num_scenes, 1), val_num, replace=False))
        train = all - val

        # Split number set
        train_scenes = list(train)
        val_scenes = list(val)

        # Token list
        train_scene_tokens = [self.scenes[_]["token"] for _ in train_scenes]
        val_scene_tokens = [self.scenes[_]["token"] for _ in val_scenes]

        # object list
        train_scene_objs = [self.scenes[_] for _ in train_scenes]
        val_scene_objs = [self.scenes[_] for _ in val_scenes]

        # sample token list
        train_sample_tokens = []
        for scene_obj in train_scene_objs:
            train_sample_tokens += self.get_sample_tokens_from_scene(scene_obj)
        val_sample_tokens = []
        for scene_obj in val_scene_objs:
            val_sample_tokens += self.get_sample_tokens_from_scene(scene_obj)

        # sample list
        train_samples = []
        for scene_obj in train_scene_objs:
            train_samples += self.get_samples_from_scene(scene_obj)
        val_samples = []
        for scene_obj in val_scene_objs:
            val_samples += self.get_samples_from_scene(scene_obj)

        return {
            "train_scene_tokens": train_scene_tokens,
            "train_scene_objs": train_scene_objs,
            "train_sample_tokens": train_sample_tokens,
            "train_samples": train_samples,
            "val_scene_tokens": val_scene_tokens,
            "val_scene_objs": val_scene_objs,
            "val_sample_tokens": val_sample_tokens,
            "val_samples": val_samples
        }

    # Get all samples in one scene
    def get_samples_from_scene(self, scene_obj):
        # Check if scene is valid
        if not self.check_scene_obj(scene_obj):
            raise ValueError

        sample_lst = []
        current_sample_token = scene_obj['first_sample_token']
        while True:
            current_sample = self.get("sample", current_sample_token)
            sample_lst.append(current_sample)
            if not current_sample["next"] == "":
                current_sample_token = current_sample["next"]
            else:
                break

        return sample_lst

    # Get all samples in one scene
    def get_sample_tokens_from_scene(self, scene_obj):
        # Check if scene is valid
        if not self.check_scene_obj(scene_obj):
            raise ValueError

        sample_token_lst = []
        current_sample_token = scene_obj['first_sample_token']
        while True:
            current_sample = self.get("sample", current_sample_token)
            sample_token_lst.append(current_sample["token"])
            if not current_sample["next"] == "":
                current_sample_token = current_sample["next"]
            else:
                break

        return sample_token_lst

    ########################
    ## Torch dataset APIs ##
    ########################
    # Add orientations to datapoints
    def get_datapoints(self, mode="train"):
        datapoints = []
        # Get orientations given dataset version
        if self.version in ["ver1", "ver3"]:
            orientations = ["front", "back"]
        elif self.version == "ver2":
            orientations = self.camera_orientations

        if mode == "train":
            samples = self.train_samples
        elif mode == "val":
            samples = self.val_samples

        for idx, sample in enumerate(samples):
            for ori in orientations:
                datapoint = [idx, ori]
                datapoints.append(datapoint)

        return datapoints

    # Get data given corresponding datapoints
    def get_data(self, datapoint, mode="train"):
        assert mode in ["train", "val"]
        # Check if datapoint is valid
        if mode == "train":
            assert datapoint in self.train_datapoints
        else:
            assert datapoint in self.val_datapoints

        sample_index = datapoint[0]
        orientation = datapoint[1]
        if mode == "train":
            sample_obj = self.train_samples[sample_index]
        else:
            sample_obj = self.val_samples[sample_index]

        lidar_data = self.get_lidar_depth_map_multi_bidirectional(sample_obj, orientation,
                                                                  num_sweeps=self.lidar_sweeps)
        radar_data = self.get_radar_depth_map_multi_bidirectional(sample_obj, orientation,
                                                                  num_sweeps=self.radar_sweeps)

        return {
            "lidar_depth": lidar_data["depth"],
            "lidar_points": lidar_data["points"],
            "lidar_depth_points": lidar_data["depth_points"],
            "radar_depth": radar_data["depth"],
            "radar_points": radar_data["points"],
            "radar_depth_points": radar_data["depth_points"],
            "radar_raw_points": radar_data["raw_points"],
            "image": lidar_data["image"]
        }

    # Get dataset length given the mode
    def get_dataset_length(self, mode="train"):
        assert mode in ["train", "val"]
        if mode == "train":
            return len(self.train_datapoints)
        else:
            return len(self.val_datapoints)

    #####################################
    ## LiDAR point cloud manipulations ##
    #####################################
    # Plot point cloud
    def get_lidar_depth_map(self, sample_obj, orientation):
        # Check the input orientation
        assert self.check_camera_orientation(orientation) == True

        # Get sensor tokens
        lidar_token = sample_obj['data']['LIDAR_TOP']
        orientation_key = self.map_camera_keyword(orientation)
        camera_front_token = sample_obj['data'][orientation_key]

        # Get projected point cloud and depth
        points, depth_points, image = self.explorer.map_pointcloud_to_image(lidar_token, camera_front_token)

        # Construct detph map
        image = np.array(image)
        depth = np.zeros(image.shape[:2])

        # Assign depth value to corresponding pixel on the image
        depth_loc = (points[:2, :].T).astype(np.int32)
        depth[depth_loc[:, 1], depth_loc[:, 0]] = depth_points

        return {
            "image": image,
            "depth": depth,
            "points": points,
            "depth_points": depth_points
        }

    # Plot lidar depth map with multiple sweeps
    def get_lidar_depth_map_multi(self, sample_obj, orientation, num_sweeps=10):
        # Check the input orientation
        assert self.check_camera_orientation(orientation) == True

        # Get sensor tokens
        lidar_token = sample_obj['data']['LIDAR_TOP']
        lidar_record = self.dataset.get("sample_data", lidar_token)

        # Get camera token
        orientation_key = self.map_camera_keyword(orientation)
        camera_front_token = sample_obj['data'][orientation_key]
        camera_record = self.dataset.get("sample_data", camera_front_token)
        im = Image.open(os.path.join(self.dataset.dataroot, camera_record['filename']))

        # Get multiple sweeps
        point_clouds, times = LidarPointCloud.from_file_multisweep(self.dataset,
                                    sample_obj, lidar_record["channel"],
                                    "LIDAR_TOP", nsweeps=num_sweeps)

        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.dataset.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        point_clouds.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        point_clouds.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = self.dataset.get('ego_pose', lidar_record['ego_pose_token'])
        point_clouds.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        point_clouds.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = self.dataset.get('ego_pose', camera_record['ego_pose_token'])
        point_clouds.translate(-np.array(poserecord['translation']))
        point_clouds.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = self.dataset.get('calibrated_sensor', camera_record['calibrated_sensor_token'])
        point_clouds.translate(-np.array(cs_record['translation']))
        point_clouds.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = point_clouds.points[2, :]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(point_clouds.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

        # Mask the depth that is not in the field of view
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        depths = depths[mask]

        # Construct detph map
        image = np.array(im)
        depth = np.zeros(image.shape[:2])

        # Assign depth value to corresponding pixel on the image
        depth_loc = (points[:2, :].T).astype(np.int32)
        depth[depth_loc[:, 1], depth_loc[:, 0]] = depths

        return {
            "image": image,
            "depth": depth,
            "points": points,
            "depth_points": depths
        }

    # Plot lidar depth map with multiple sweeps (accumulate from both past and future)
    def get_lidar_depth_map_multi_bidirectional(self, sample_obj, orientation, num_sweeps=3):
        # Check the input orientation
        assert self.check_camera_orientation(orientation) == True

        # Get sensor tokens
        lidar_token = sample_obj['data']['LIDAR_TOP']
        lidar_record = self.dataset.get("sample_data", lidar_token)

        # Get camera token
        orientation_key = self.map_camera_keyword(orientation)
        camera_front_token = sample_obj['data'][orientation_key]
        camera_record = self.dataset.get("sample_data", camera_front_token)
        im = Image.open(os.path.join(self.dataset.dataroot, camera_record['filename']))

        # Divide num_sweeps into two parts
        next_sweeps = math.floor((num_sweeps - 1) / 2)
        prev_sweeps = math.ceil((num_sweeps - 1) / 2)

        # Get current point cloud
        point_clouds, times = LidarPointCloud.from_file_multisweep(self.dataset, sample_obj,
                                                            lidar_record["channel"], "LIDAR_TOP",
                                                            nsweeps=1, min_distance=0.)

        # Get prev point cloud
        if prev_sweeps > 0:
            prev_point_clouds, prev_times = self.accumulate_lidar(sample_obj, "prev", prev_sweeps, 1.0,
                                                                  orientation, image_size=im.size)
            point_clouds.points = np.hstack((point_clouds.points, prev_point_clouds.points))
            times = np.hstack((times, prev_times))

        # Get next point cloud
        if next_sweeps > 0:
            next_point_clouds, next_times = self.accumulate_lidar(sample_obj, "next", next_sweeps, 1.0,
                                                                  orientation, image_size=im.size)
            point_clouds.points = np.hstack((point_clouds.points, next_point_clouds.points))
            times = np.hstack((times, next_times))

        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.dataset.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        point_clouds.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        point_clouds.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = self.dataset.get('ego_pose', lidar_record['ego_pose_token'])
        point_clouds.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        point_clouds.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = self.dataset.get('ego_pose', camera_record['ego_pose_token'])
        point_clouds.translate(-np.array(poserecord['translation']))
        point_clouds.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = self.dataset.get('calibrated_sensor', camera_record['calibrated_sensor_token'])
        point_clouds.translate(-np.array(cs_record['translation']))
        point_clouds.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = point_clouds.points[2, :]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        # ToDo: Add options of projecting to a low resolution camera
        # Adjust image size directly
        if cfg.scaling is True:
            im = im.resize((int(im.size[0] * cfg.scale_factor), int(im.size[1] * cfg.scale_factor)),
                           Image.BILINEAR)

        # Adjust camera intrinsics for depth projection
        if cfg.scaling is True:
            intrinsics = np.concatenate((
                (np.array(cs_record['camera_intrinsic'][0]) * cfg.scale_factor)[None, ...],
                (np.array(cs_record['camera_intrinsic'][1]) * cfg.scale_factor)[None, ...],
                (np.array(cs_record['camera_intrinsic'][2]))[None, ...]
            ), axis=0)

        else:
            intrinsics = np.array(cs_record['camera_intrinsic'])

        points = view_points(point_clouds.points[:3, :], intrinsics, normalize=True)
        # ipdb.set_trace()

        # Mask the depth that is not in the field of view
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        # Check vertical field of view
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        # Check horizontal field of view
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        depths = depths[mask]

        # Construct detph map
        image = np.array(im)
        depth = np.zeros(image.shape[:2])

        # Assign depth value to corresponding pixel on the image
        depth_loc = (points[:2, :].T).astype(np.int32)
        depth[depth_loc[:, 1], depth_loc[:, 0]] = depths

        return {
            "image": image,
            "depth": depth,
            "points": points,
            "depth_points": depths
        }

    # ToDo: remove points that are not supposed to be inside field of view of each time steps
    # Accumulate lidar point given the direction
    def accumulate_lidar(self, sample_obj, direction="prev", num_sweeps=2, min_distance=1.0,
                         cam_orientation="front", image_size=None):
        """
        :param sample_obj: object of the reference sample
        :param direction: "prev" or "next" => direction to accumulate lidar points
        :param num_sweeps: total number of lidar sweeps to accumulate
        :param min_distance: minimum distance to keep for close points
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        assert direction in ["prev", "next"]
        assert image_size is not None

        # Initialize point obj
        points = np.zeros([LidarPointCloud.nbr_dims(), 0])
        all_points = LidarPointCloud(points)
        all_times = np.zeros((1, 0))

        # Get lidar sensor info
        ref_lidar_token = sample_obj['data']['LIDAR_TOP']
        ref_lidar_obj = self.dataset.get("sample_data", ref_lidar_token)
        ref_pose_obj = self.dataset.get('ego_pose', ref_lidar_obj['ego_pose_token'])
        ref_cs_obj = self.dataset.get('calibrated_sensor', ref_lidar_obj['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_lidar_obj['timestamp']

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_obj['translation'], Quaternion(ref_cs_obj['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_obj['translation'], Quaternion(ref_pose_obj['rotation']),
                                           inverse=True)

        # Aggregate previous sweeps.
        sample_data_token = sample_obj['data']["LIDAR_TOP"]
        cam_sample_data_token = sample_obj['data'][self.map_camera_keyword(cam_orientation)]

        # Start from the "first previous" or "first next" one
        # ToDo: check if previous one exists
        first_token = self.dataset.get('sample_data', sample_data_token)[direction]
        cam_first_token = self.dataset.get('sample_data', cam_sample_data_token)[direction]

        if first_token == "":
            return all_points, all_times
        else:
            current_lidar_obj = self.dataset.get('sample_data', first_token)
            try:
                current_cam_obj = self.dataset.get('sample_data', cam_first_token)
            except:
                raise ValueError("Possible reason... LiDAR has previous record does not mean that camera will have..")

        # Iterate through past or future
        for _ in range(num_sweeps):
            # Load up the pointcloud.
            current_pc = LidarPointCloud.from_file(os.path.join(self.dataset.dataroot, current_lidar_obj['filename']))

            # Get the transformation from sensor to world in this time step
            current_pose_rec = self.dataset.get('ego_pose', current_lidar_obj['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = self.dataset.get('calibrated_sensor', current_lidar_obj['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # ToDo: Check point clouds in the current field of view
            # ToDo: What about the synchronization between LiDAR and Camera (ignore that currently)
            # [Lidar => Car => Camera => check => Car => Lidar]
            # This part is just filtering out lidar points not in the camera field of view of the same time step
            calib_cam_obj = self.dataset.get('calibrated_sensor', current_cam_obj['calibrated_sensor_token'])
            car_to_camera = transform_matrix(calib_cam_obj['translation'], Quaternion(calib_cam_obj["rotation"]),
                                         inverse=True)
            lidar_to_cam = reduce(np.dot, [car_to_camera, car_from_current])
            cam_to_lidar = reduce(np.dot, [car_from_current.T, car_to_camera.T])
            current_pc = self.check_point_clouds_single_frame(current_pc,
                                                              lidar_to_cam=lidar_to_cam,
                                                              cam_to_lidar=cam_to_lidar,
                                                              cam_intrinsics=calib_cam_obj['camera_intrinsic'],
                                                              image_size=image_size)

            # Fuse four transformation matrices into one and perform transform.
            # current_sensor => current_car => global => ref car => ref_sensor
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * current_lidar_obj['timestamp']  # positive difference
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            all_times = np.hstack((all_times, times))

            # Merge with key pc.
            all_points.points = np.hstack((all_points.points, current_pc.points))

            # Abort if there are no previous or next sweeps.
            if current_lidar_obj[direction] == '':
                break
            else:
                current_lidar_obj = self.dataset.get('sample_data', current_lidar_obj[direction])

        return all_points, all_times

    def check_point_clouds_single_frame(self, input_points, lidar_to_cam, cam_to_lidar, cam_intrinsics, image_size):
        # Transform all the points from LiDAR to Camera
        input_points.transform(lidar_to_cam)

        # Get the mask
        projected_points = view_points(input_points.points[:3, :], np.array(cam_intrinsics), normalize=True)
        projected_depths = input_points.points[2, :]

        # Mask the depth that is not in the field of view
        mask = np.ones(projected_depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, projected_depths > 0)

        # Check vertical field of view
        # ToDo: We crop 5% of the bottom part (ground and car removal)
        mask = np.logical_and(mask, projected_points[0, :] > int(image_size[0]*0.05))
        mask = np.logical_and(mask, projected_points[0, :] < image_size[0] - int(image_size[0]*0.05))

        # Check horizontal field of view
        mask = np.logical_and(mask, projected_points[1, :] > 1)
        mask = np.logical_and(mask, projected_points[1, :] < image_size[1] - 1)
        input_points.points = input_points.points[:, mask]

        # Transform all points from Camera to LiDAR
        input_points.transform(cam_to_lidar)

        return input_points

    #####################################
    ## Radar point cloud manipulations ##
    #####################################
    # Plot point cloud using radar
    # ToDo: make the concatenation of multiple point clouds
    def get_radar_depth_map(self, sample_obj, camera_orientation='front'):
        # Check the input orientation
        assert self.check_camera_orientation(camera_orientation) == True

        # Get sensor tokens
        orientation_key = self.map_camera_keyword(camera_orientation)
        camera_front_token = sample_obj['data'][orientation_key]

        point_list = []
        depth_point_list = []
        # Get projected point cloud and depth
        for ori in self.radar_orientations:
            # Iterate through radar orientations
            orientation_key = self.map_radar_keyword(ori)
            radar_token = sample_obj['data'][orientation_key]
            points, depth_points, image = self.explorer.map_pointcloud_to_image(radar_token, camera_front_token)
            point_list.append(points)
            depth_point_list.append(depth_points)

        points = np.concatenate(tuple(point_list), axis=-1)
        depth_points = np.concatenate(tuple(depth_point_list), axis=-1)

        # Construct detph map
        image = np.array(image)
        depth = np.zeros(image.shape[:2])

        # Assign depth value to corresponding pixel on the image
        depth_loc = (points[:2, :].T).astype(np.int32)
        depth[depth_loc[:, 1], depth_loc[:, 0]] = depth_points

        return {
            "image": image,
            "depth": depth,
            "points": points,
            "depth_points": depth_points
        }

    # Accumulate radar points from multiple time steps
    def get_radar_depth_map_multi(self, sample_obj, orientation='front', num_sweeps=10):
        # Check the input orientation
        assert self.check_camera_orientation(orientation) == True

        # Get radar tokens
        radar_tokens = [sample_obj['data'][_] for _ in self.radar_keywords]
        radar_records = [self.get("sample_data", _) for _ in radar_tokens]

        # Get camera token
        orientation_key = self.map_camera_keyword(orientation)
        camera_front_token = sample_obj['data'][orientation_key]
        camera_record = self.dataset.get("sample_data", camera_front_token)
        im = Image.open(os.path.join(self.dataset.dataroot, camera_record['filename']))

        # Get multiple sweeps
        point_cloud_list = []
        time_list = []
        for i in range(len(radar_records)):
            radar_record = radar_records[i]
            point_clouds, times = RadarPointCloud.from_file_multisweep(self.dataset,
                                                            sample_obj, radar_record["channel"],
                                                            "LIDAR_TOP", nsweeps=num_sweeps)
            point_cloud_list.append(point_clouds.points)
            time_list.append(time_list)

        # ipdb.set_trace()

    # Accumulate radar points from bidirectional (accumulate both past and future
    def get_radar_depth_map_multi_bidirectional(self, sample_obj, orientation="front", num_sweeps=3):
        # Check the input orientation
        assert self.check_camera_orientation(orientation) == True

        # Get corresponding sensor tokens
        related_radar_keywords = [self.map_radar_keyword(_) \
                                  for _ in self.get_radar_orientation_from_camera(orientation)]
        radar_tokens = [sample_obj['data'][_] for _ in related_radar_keywords]
        radar_records = [self.get("sample_data", _) for _ in radar_tokens]

        # Get camera token
        orientation_key = self.map_camera_keyword(orientation)
        camera_front_token = sample_obj['data'][orientation_key]
        camera_record = self.dataset.get("sample_data", camera_front_token)
        im = Image.open(os.path.join(self.dataset.dataroot, camera_record['filename']))

        # Also get lidar sensor (because we map to lidar frame
        lidar_token = sample_obj['data']['LIDAR_TOP']
        lidar_record = self.get("sample_data", lidar_token)

        # Divide num_sweeps into two parts
        next_sweeps = math.floor((num_sweeps - 1) / 2)
        prev_sweeps = math.ceil((num_sweeps - 1) / 2)

        # Get current point cloud
        point_cloud_list = []
        time_list = []
        for i in range(len(radar_records)):
            # Original version: all radar points are mapped to LIDAR_TOP
            radar_record = radar_records[i]
            point_clouds, times = RadarPointCloud.from_file_multisweep(self.dataset, sample_obj,
                                                                       radar_record["channel"], "LIDAR_TOP",
                                                                       nsweeps=1)
            # ###################################
            # ## Further perform yaw filtering ##
            # ###################################
            # # Map to radar frame first
            # point_clouds, times = RadarPointCloud.from_file_multisweep(self.dataset, sample_obj,
            #                                                            radar_record["channel"], radar_record["channel"],
            #                                                            nsweeps=1)
            # # Filter the points by field of view
            # point_clouds.points = self.radar_filter_valid_angles(point_clouds.points.transpose(1, 0), 60, True)
            # # Transform to ego frame
            # radar_calibrated_sensor = self.get("calibrated_sensor", radar_record["calibrated_sensor_token"])
            # point_clouds.rotate(Quaternion(radar_calibrated_sensor["rotation"]).rotation_matrix)
            # point_clouds.translate(np.array(radar_calibrated_sensor["translation"]))
            # # Transform to lidar top
            # lidar_calibrated_sensor = self.get("calibrated_sensor", lidar_record["calibrated_sensor_token"])
            # point_clouds.translate(-np.array(lidar_calibrated_sensor["translation"]))
            # point_clouds.rotate(Quaternion(lidar_calibrated_sensor["rotation"]).rotation_matrix.T)
            # ipdb.set_trace()

            point_cloud_list.append(point_clouds.points)
            time_list.append(times)

        # ipdb.set_trace()

        # Stack all the points along the second dimension
        point_clouds = RadarPointCloud(np.concatenate(tuple(point_cloud_list), axis=1))
        times = np.concatenate(tuple(time_list), axis=1)

        # Get prev point cloud
        if prev_sweeps > 0:
            prev_point_clouds, prev_times = self.accumulate_radar(sample_obj, "prev", prev_sweeps, 1.0,
                                                                  orientation, image_size=im.size)
            point_clouds.points = np.hstack((point_clouds.points, prev_point_clouds.points))
            times = np.hstack((times, prev_times))

        # Get next point cloud
        if next_sweeps > 0:
            next_point_clouds, next_times = self.accumulate_radar(sample_obj, "next", next_sweeps, 1.0,
                                                                  orientation, image_size=im.size)
            point_clouds.points = np.hstack((point_clouds.points, next_point_clouds.points))
            times = np.hstack((times, next_times))

        # ToDo: for the accumulated ones, should we transform them to the lidar frame?

        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        point_clouds.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        point_clouds.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = self.dataset.get('ego_pose', lidar_record['ego_pose_token'])
        point_clouds.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        point_clouds.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = self.dataset.get('ego_pose', camera_record['ego_pose_token'])
        point_clouds.translate(-np.array(poserecord['translation']))
        point_clouds.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = self.dataset.get('calibrated_sensor', camera_record['calibrated_sensor_token'])
        point_clouds.translate(-np.array(cs_record['translation']))
        point_clouds.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = point_clouds.points[2, :]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        # ToDo: Add options of projecting to a low resolution camera
        # Adjust image size directly
        if cfg.scaling is True:
            im = im.resize((int(im.size[0] * cfg.scale_factor), int(im.size[1] * cfg.scale_factor)),
                           Image.BILINEAR)

        # Adjust camera intrinsics for depth projection
        if cfg.scaling is True:
            intrinsics = np.concatenate((
                (np.array(cs_record['camera_intrinsic'][0]) * cfg.scale_factor)[None, ...],
                (np.array(cs_record['camera_intrinsic'][1]) * cfg.scale_factor)[None, ...],
                (np.array(cs_record['camera_intrinsic'][2]))[None, ...]
            ), axis=0)

        else:
            intrinsics = np.array(cs_record['camera_intrinsic'])

        points = view_points(point_clouds.points[:3, :], intrinsics, normalize=True)

        # Mask the depth that is not in the field of view
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        # Check vertical field of view
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        # Check horizontal field of view
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        depths = depths[mask]

        # Construct detph map
        image = np.array(im)
        depth = np.zeros(image.shape[:2])

        # Assign depth value to corresponding pixel on the image
        depth_loc = (points[:2, :].T).astype(np.int32)
        depth[depth_loc[:, 1], depth_loc[:, 0]] = depths

        return {
            "image": image,
            "depth": depth,
            "points": points,
            "depth_points": depths,
            "raw_points": point_clouds.points[:, mask]
        }

    def get_radar_depth_map_multi_bidirectional_experimental(self, sample_obj, orientation="front", num_sweeps=3):
        # Check the input orientation
        assert self.check_camera_orientation(orientation) == True

        # Get corresponding sensor tokens
        related_radar_keywords = [self.map_radar_keyword(_) \
                                  for _ in self.get_radar_orientation_from_camera(orientation)]
        radar_tokens = [sample_obj['data'][_] for _ in related_radar_keywords]
        radar_records = [self.get("sample_data", _) for _ in radar_tokens]

        # Get camera token
        orientation_key = self.map_camera_keyword(orientation)
        camera_front_token = sample_obj['data'][orientation_key]
        camera_record = self.dataset.get("sample_data", camera_front_token)
        im = Image.open(os.path.join(self.dataset.dataroot, camera_record['filename']))

        # Also get lidar sensor (because we map to lidar frame
        lidar_token = sample_obj['data']['LIDAR_TOP']
        lidar_record = self.get("sample_data", lidar_token)

        # Divide num_sweeps into two parts
        next_sweeps = math.floor((num_sweeps - 1) / 2)
        prev_sweeps = math.ceil((num_sweeps - 1) / 2)

        # Get current point cloud
        point_cloud_list = []
        time_list = []
        for i in range(len(radar_records)):
            radar_record = radar_records[i]
            point_clouds, times = RadarPointCloud.from_file_multisweep(self.dataset, sample_obj,
                                                                       radar_record["channel"], "LIDAR_TOP",
                                                                       nsweeps=1)
            # ipdb.set_trace()
            point_cloud_list.append(point_clouds.points)
            time_list.append(times)

        # ipdb.set_trace()

        # Stack all the points along the second dimension
        point_clouds = RadarPointCloud(np.concatenate(tuple(point_cloud_list), axis=1))
        times = np.concatenate(tuple(time_list), axis=1)

        # Get prev point cloud
        if prev_sweeps > 0:
            prev_point_clouds, prev_times = self.accumulate_radar(sample_obj, "prev", prev_sweeps, 1.0,
                                                                  orientation, image_size=im.size)
            point_clouds.points = np.hstack((point_clouds.points, prev_point_clouds.points))
            times = np.hstack((times, prev_times))

        # Get next point cloud
        if next_sweeps > 0:
            next_point_clouds, next_times = self.accumulate_radar(sample_obj, "next", next_sweeps, 1.0,
                                                                  orientation, image_size=im.size)
            point_clouds.points = np.hstack((point_clouds.points, next_point_clouds.points))
            times = np.hstack((times, next_times))

        # ToDo: for the accumulated ones, should we transform them to the lidar frame?

        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = self.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        point_clouds.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        point_clouds.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = self.dataset.get('ego_pose', lidar_record['ego_pose_token'])
        point_clouds.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        point_clouds.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = self.dataset.get('ego_pose', camera_record['ego_pose_token'])
        point_clouds.translate(-np.array(poserecord['translation']))
        point_clouds.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = self.dataset.get('calibrated_sensor', camera_record['calibrated_sensor_token'])
        point_clouds.translate(-np.array(cs_record['translation']))
        point_clouds.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = point_clouds.points[2, :]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        # ToDo: Add options of projecting to a low resolution camera
        # Adjust image size directly
        if cfg.scaling is True:
            im = im.resize((int(im.size[0] * cfg.scale_factor), int(im.size[1] * cfg.scale_factor)),
                           Image.BILINEAR)

        # Adjust camera intrinsics for depth projection
        if cfg.scaling is True:
            intrinsics = np.concatenate((
                (np.array(cs_record['camera_intrinsic'][0]) * cfg.scale_factor)[None, ...],
                (np.array(cs_record['camera_intrinsic'][1]) * cfg.scale_factor)[None, ...],
                (np.array(cs_record['camera_intrinsic'][2]))[None, ...]
            ), axis=0)

        else:
            intrinsics = np.array(cs_record['camera_intrinsic'])

        points = view_points(point_clouds.points[:3, :], intrinsics, normalize=True)

        # Mask the depth that is not in the field of view
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        # Check vertical field of view
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        # Check horizontal field of view
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        depths = depths[mask]

        # Construct detph map
        image = np.array(im)
        depth = np.zeros(image.shape[:2])

        # Assign depth value to corresponding pixel on the image
        depth_loc = (points[:2, :].T).astype(np.int32)
        depth[depth_loc[:, 1], depth_loc[:, 0]] = depths

        ########### Add back traverse map ##############
        points_index = np.arange(0, points.shape[1], 1)
        depth_index = - np.ones(image.shape[:2])
        depth_index[depth_loc[:, 1], depth_loc[:, 0]] = points_index

        return {
            "image": image,
            "depth": depth,
            "points": points,
            "depth_points": depths,
            "raw_points": point_clouds.points[:, mask],
            "index_map": depth_index
        }

    # Accumulate radar points given the direction in ["prev", "next"]
    def accumulate_radar(self, sample_obj, direction="prev", num_sweeps=2, min_distance=0.5,
                         cam_orientation="front", image_size=None):
        """
        :param sample_obj: object of the reference sample
        :param direction: "prev" or "next" => direction to accumulate radar points
        :param num_sweeps: total number of radar sweeps to accumulate
        :param min_distance: minimum distance to keep for close points
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        assert direction in ["prev", "next"]
        assert image_size is not None

        # Initialize point obj
        points = np.zeros([RadarPointCloud.nbr_dims(), 0])
        all_points = RadarPointCloud(points)
        all_times = np.zeros((1, 0))

        # Get reference lidar sensor info => because we want to accumulate to lidar frame
        ref_lidar_token = sample_obj['data']['LIDAR_TOP']
        ref_lidar_obj = self.get("sample_data", ref_lidar_token)
        ref_pose_obj = self.get('ego_pose', ref_lidar_obj['ego_pose_token'])
        ref_cs_obj = self.get('calibrated_sensor', ref_lidar_obj['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_lidar_obj['timestamp']

        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_obj['translation'], Quaternion(ref_cs_obj['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_obj['translation'], Quaternion(ref_pose_obj['rotation']),
                                           inverse=True)

        # Aggregate previous sweeps.
        # ToDo: should we do it individually or jointly => each radar gets its prev or...
        # First, we get the related radar directions
        related_radar_keywords = [self.map_radar_keyword(_) \
                                  for _ in self.get_radar_orientation_from_camera(cam_orientation)]
        radar_sample_data_tokens = [sample_obj['data'][_] for _ in related_radar_keywords]
        cam_sample_data_token = sample_obj['data'][self.map_camera_keyword(cam_orientation)]

        # Start from the "first previous" or "first next" records
        first_tokens = [self.get("sample_data", _)[direction] for _ in radar_sample_data_tokens]
        cam_first_token = self.get('sample_data', cam_sample_data_token)[direction]

        if "" in first_tokens:
            return all_points, all_times
        else:
            current_radar_objs = [self.get('sample_data', _) for _ in first_tokens]
            try:
                current_cam_obj = self.dataset.get('sample_data', cam_first_token)
            except:
                raise ValueError("Possible reason... LiDAR has previous record does not mean that camera will have..")

        # Iterate through past or future and accumulate each sensor separately
        for current_radar_obj in current_radar_objs:
            points, times = self.accumulate_radar_single(current_radar_obj, num_sweeps, direction, ref_from_car,
                                                         car_from_global, ref_time, min_distance)
            all_points.points = np.concatenate((all_points.points, points.points), axis=1)
            all_times = np.concatenate((all_times, times), axis=1)

        return all_points, all_times

    def accumulate_radar_single(self, current_radar_obj, num_sweeps, direction, ref_from_car, car_from_global, ref_time,
                                min_distance):
        # Declare accumulation container
        points = np.zeros([RadarPointCloud.nbr_dims(), 0])
        output_points = RadarPointCloud(points)
        output_times = np.zeros((1, 0))

        # Iterate through past or future
        for _ in range(num_sweeps):
            # Load the pointcloud
            current_pc = RadarPointCloud.from_file(os.path.join(self.dataset.dataroot, current_radar_obj['filename']))

            # Get the transformation from sensor to world in this time step
            current_pose_rec = self.get('ego_pose', current_radar_obj['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = self.dataset.get('calibrated_sensor', current_radar_obj['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # ToDo: Check point clouds in the current field of view
            # ToDo: What about the synchronization between LiDAR and Camera (ignore that currently)
            # Fuse four transformation matrices into one and perform transform.
            # current_sensor => current_car => global => ref car => ref_sensor
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Remove close points and add timevector.
            current_pc.remove_close(min_distance)
            time_lag = ref_time - 1e-6 * current_radar_obj['timestamp']  # positive difference
            times = time_lag * np.ones((1, current_pc.nbr_points()))
            output_times = np.concatenate((output_times, times), axis=1)

            # Merge all the pc
            output_points.points = np.concatenate((output_points.points, current_pc.points), axis=1)

            # Abort if there are no previous or next sweeps.
            if current_radar_obj[direction] == "":
                break
            else:
                current_radar_obj = self.get('sample_data', current_radar_obj[direction])

        return output_points, output_times

    # Get related radar orientations given camera orientation
    def get_radar_orientation_from_camera(self, cam_orientation):
        assert self.check_camera_orientation(cam_orientation) == True

        # Expand all the possible combinations
        if cam_orientation == "front":
            output_orientations = ["front", "front_right", "front_left"]
        elif cam_orientation == "front_right":
            output_orientations = ["front_left", "front", "front_right"]
        elif cam_orientation == "front_left":
            output_orientations = ['front', 'front_right', 'front_left']
        elif cam_orientation == "back_right":
            output_orientations = ['front_right', 'back_right', 'back_left']
        elif cam_orientation == "back_left":
            output_orientations = ['front_left', 'back_right', 'back_left']
        elif cam_orientation == "back":
            output_orientations = ['back_right', 'back_left']
        else:
            raise ValueError("[Error] Should not get here...")

        return output_orientations

    # Method to filter out radar returns outside the field of view.
    @staticmethod
    def radar_filter_valid_angles(points: np.ndarray,
                                  max_abs_angle: float = 60,
                                  debug: bool = False) -> np.ndarray:
        """
        # Filter points outside the radar FOV (in the xy plane).
        :param points: <np.float: 4, num_points>.
        :param max_abs_angle: The maximum absolute angle above which radar returns will be removed.
        :param debug: Whether to print debugging messages about the angles and the number of points.
        :return: The filtered pointcloud.
        """
        # We use the x and y values to compute the yaw angle.
        # The convoluted math rotates the "origin" from pointing right to pointing forward and flips it.
        angles = ((90 - (np.arctan2(points[:, 0], points[:, 1]) / np.pi * 180)) + 180) % 360 - 180
        if debug:
            print(angles.astype(np.int32))

        # Filter out the points that have invalid angles.
        valid = np.abs(angles) <= max_abs_angle
        pointcount_before = len(angles)
        pointcount_after = valid.sum()
        points = points[valid, :]
        if debug:
            print('Total points: %d, valid points: %d' % (pointcount_before, pointcount_after))
            min_angle = np.max(np.abs(angles))
            print('Minimum absolute angle was: %f' % min_angle)

        return points.transpose(1, 0)

    #########################
    ## Some checking codes ##
    #########################
    # Check if the input orientation is in the table
    def check_camera_orientation(self, input_orientation):
        if input_orientation in self.camera_orientations:
            return True
        else:
            print("[Error] the orientation is not in", self.camera_orientations)
            raise ValueError("Unknown input orientation!")

    # Check if the input radar orientation is in the table
    def check_radar_orientation(self, input_orientation):
        if input_orientation in self.radar_orientations:
            return True
        else:
            print("[Error] the orientation is not in", self.radar_orientations)
            raise ValueError("Unknown input orientation!")

    # Map orientation to keyword in nuscenes
    def map_camera_keyword(self, input_orientation):
        return self.orientation_to_camera[input_orientation]

    # Map orientation to keyword in nuscenes
    def map_radar_keyword(self, input_orientation):
        return self.orientation_to_radar[input_orientation]

    # Check scene
    def check_scene_obj(self, scene_obj):
        if scene_obj in self.scenes:
            return True
        else:
            raise ValueError("[Error] Scene object is not in the dataset.")


# main function for running some debugging
if __name__ == "__main__":
    dataset = Nuscenes_dataset(mode="mini")
    datapoint = dataset.train_datapoints[10]
    data = dataset.get_data(datapoint, mode="train")
    ipdb.set_trace()

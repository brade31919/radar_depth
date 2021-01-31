from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)

import sys
sys.path.append("../")

import os
import numpy as np
import matplotlib.pyplot as plt
from dataset.nuscenes_dataset import Nuscenes_dataset
from config.config_nuscenes import config_nuscenes as cfg


# Use more loose depth threshold for points with larger depth value
def sid_depth_thresh(input_depth):
    alpha = 5
    beta = 16
    K = 100

    depth_thresh = np.exp(((input_depth * np.log(beta / alpha)) / K) + np.log(alpha))

    return depth_thresh


# Use more strict distance threshold for points with larger depth value
def sid_dist_thresh(input_depth):
    alpha = 14
    beta = 4
    K = 100

    dist_thresh = np.exp(((input_depth * np.log(beta / alpha)) / K) + np.log(alpha))

    return dist_thresh


# Select the depth value of the topk candidates from ndarray
def select_topk_depth(depth, topk_idx):
    # Select topk index and topk value
    point_count = topk_idx.shape[0]

    topk_depth_lst = []
    for i in range(point_count):
        # ipdb.set_trace()
        topk_depth = depth[topk_idx[i, :]]
        topk_depth_lst.append(topk_depth)

    return np.asarray(topk_depth_lst)


# Check topk candidates using depth threshold
def check_valid_depth(depth_dist, dist_valid_count, depth_value):
    point_count = depth_dist.shape[0]

    # ipdb.set_trace()
    # 0 => invalid, 1 => valid, 2 => unknown
    valid_labels = np.zeros([point_count, 1])
    # Iterate through all the radar points
    for i in range(point_count):
        # ipdb.set_trace()
        depth_thresh_new = sid_depth_thresh(depth_value[i, :dist_valid_count[i, 0]])
        depth_valid_count = np.sum((depth_dist[i, :dist_valid_count[i, 0]] < depth_thresh_new).astype(np.int16))
        # ipdb.set_trace()
        if dist_valid_count[i, 0] == 0:
            valid_labels[i, 0] = 2
        elif depth_valid_count >= np.ceil(dist_valid_count[i, 0] / 2):
            valid_labels[i, 0] = 1

    return valid_labels


# Filter radar points using the groundtruth LiDAR points
def filter_radar_points_gt(radar_points, radar_depth_points, lidar_points, lidar_depth_points):
    # Find k nearest neighbors whithin distance threshold
    k = 3
    dist_thresh = 10

    # Fetch only the x, y coord
    radar_points = radar_points[:2, :].transpose(1, 0)
    lidar_points = lidar_points[:2, :].transpose(1, 0)

    # Mask out points > 80m
    # radar_mask = radar_depth_points < 80.
    # radar_depth_points = radar_depth_points[radar_mask]
    # radar_points = radar_points[radar_mask, :]

    radar_points_exp = np.expand_dims(radar_points, axis=1)
    lidar_points_exp = np.expand_dims(lidar_points, axis=0)

    dist = np.sqrt(np.sum((radar_points_exp - lidar_points_exp) ** 2, axis=-1))

    # Fetch the topk index
    dist_topk_index = np.argsort(dist)[:, :k][..., None]
    # Get dist topk value
    dist_topk_val = np.sort(dist, axis=-1)[:, :k]
    # Get depth topk depth value
    depth_topk_val = np.squeeze(select_topk_depth(lidar_depth_points, dist_topk_index))

    # Get depth-aware dist thresh
    dist_thresh_sid = sid_dist_thresh(depth_topk_val)
    dist_valid_count = np.sum((dist_topk_val <= dist_thresh_sid).astype(np.int16), axis=-1)[..., None]

    # print(sid_dist_thresh(depth_topk_val))
    depth_dist = radar_depth_points[..., None] - depth_topk_val
    valid_labels = check_valid_depth(depth_dist, dist_valid_count, depth_topk_val)

    # ipdb.set_trace()
    # Perform the filtering
    valid_mask_final = np.squeeze(valid_labels > 0)
    radar_points_filtered = radar_points[valid_mask_final, :].transpose(1, 0)
    radar_depth_points_filtered = radar_depth_points[valid_mask_final]

    return {
        'valid_labels': valid_labels,
        'valid_mask': valid_mask_final,
        'radar_points': radar_points_filtered,
        'radar_depth': radar_depth_points_filtered
    }


# Filter radar points using the groundtruth LiDAR points
def filter_radar_points_analysis(radar_points, radar_depth_points, lidar_points, lidar_depth_points):
    # Find k nearest neighbors whithin distance threshold
    k = 3
    dist_thresh = 10

    # Fetch only the x, y coord
    radar_points = radar_points[:2, :].transpose(1, 0)
    lidar_points = lidar_points[:2, :].transpose(1, 0)

    radar_points_exp = np.expand_dims(radar_points, axis=1)
    lidar_points_exp = np.expand_dims(lidar_points, axis=0)

    dist = np.sqrt(np.sum((radar_points_exp - lidar_points_exp) ** 2, axis=-1))

    # Fetch the topk index
    dist_topk_index = np.argsort(dist)[:, :k][..., None]
    # Get dist topk value
    dist_topk_val = np.sort(dist, axis=-1)[:, :k]
    # Get depth topk depth value
    depth_topk_val = np.squeeze(select_topk_depth(lidar_depth_points, dist_topk_index))

    # ipdb.set_trace()

    # Get depth-aware dist thresh
    dist_thresh_sid = sid_dist_thresh(depth_topk_val)
    dist_valid_count = np.sum((dist_topk_val <= dist_thresh_sid).astype(np.int16), axis=-1)[..., None]

    # print(sid_dist_thresh(depth_topk_val))
    depth_dist = radar_depth_points[..., None] - depth_topk_val
    valid_labels = check_valid_depth(depth_dist, dist_valid_count, depth_topk_val)

    # ipdb.set_trace()
    # Perform the filtering
    valid_mask_final = np.squeeze(valid_labels > 0)
    radar_points_filtered = radar_points[valid_mask_final, :].transpose(1, 0)
    radar_depth_points_filtered = radar_depth_points[valid_mask_final]

    ###########################
    ## Compute some analysis ##
    ###########################
    # Compute inconsistencies using top-1 nearest neighbor
    # ipdb.set_trace()
    depth_top1_val = depth_topk_val[:, 0][..., None]
    depth_inconsist_raw = radar_depth_points[..., None] - depth_top1_val

    # depth_inconsist = depth_inconsist_raw[np.squeeze(valid_labels < 2), :]
    # depth_inconsist_filtered = depth_inconsist_raw[np.squeeze(valid_labels == 1), :]
    # ipdb.set_trace()

    return {
        'valid_labels': valid_labels,
        'valid_mask': valid_mask_final,
        'radar_points': radar_points_filtered,
        'radar_depth': radar_depth_points_filtered,
        'depth_top1_val': depth_top1_val,
        'depth_inconsist_raw': depth_inconsist_raw
    }


# Filter radar points using the groundtruth LiDAR points
def filter_radar_points_classify(radar_points, radar_depth_points, radar_raw_points, classifyer=None):
    # Find k nearest neighbors whithin distance threshold
    assert classifyer is not None

    # Fetch only the x, y coord
    radar_points = radar_points[:2, :].transpose(1, 0)
    lidar_points = lidar_points[:2, :].transpose(1, 0)

    # Mask out points > 80m
    # radar_mask = radar_depth_points < 80.
    # radar_depth_points = radar_depth_points[radar_mask]
    # radar_points = radar_points[radar_mask, :]

    radar_points_exp = np.expand_dims(radar_points, axis=1)
    lidar_points_exp = np.expand_dims(lidar_points, axis=0)

    dist = np.sqrt(np.sum((radar_points_exp - lidar_points_exp) ** 2, axis=-1))

    # Fetch the topk index
    dist_topk_index = np.argsort(dist)[:, :k][..., None]
    # Get dist topk value
    dist_topk_val = np.sort(dist, axis=-1)[:, :k]
    # Get depth topk depth value
    depth_topk_val = np.squeeze(select_topk_depth(lidar_depth_points, dist_topk_index))

    # Get depth-aware dist thresh
    dist_thresh_sid = sid_dist_thresh(depth_topk_val)
    dist_valid_count = np.sum((dist_topk_val <= dist_thresh_sid).astype(np.int16), axis=-1)[..., None]

    # print(sid_dist_thresh(depth_topk_val))
    depth_dist = radar_depth_points[..., None] - depth_topk_val
    valid_labels = check_valid_depth(depth_dist, dist_valid_count, depth_topk_val)

    # ipdb.set_trace()
    # Perform the filtering
    valid_mask_final = np.squeeze(valid_labels > 0)
    radar_points_filtered = radar_points[valid_mask_final, :].transpose(1, 0)
    radar_depth_points_filtered = radar_depth_points[valid_mask_final]

    return {
        'valid_labels': valid_labels,
        'valid_mask': valid_mask_final,
        'radar_points': radar_points_filtered,
        'radar_depth': radar_depth_points_filtered
    }


# Plot LiDAR depth
def plot_lidar_depth(image, points, depth_points, vmin=0., vmax=80.):
    # Plot depth
    fig = plt.figure(figsize=(10,6))
    ax = plt.gca()
    plt.imshow(image)
    im = plt.scatter(points[0, :], points[1, :], c=depth_points, s=3, vmin=vmin, vmax=vmax, cmap="jet")
    fig.colorbar(im, ax=ax, cmap="jet", pad=0.01)
    plt.show()


# Plot Radar depth
def plot_radar_depth(image, points, depth_points, vmin=0., vmax=80.):
    # Plot depth
    fig = plt.figure(figsize=(10,6))
    ax = plt.gca()
    plt.imshow(image)
    im = plt.scatter(points[0, :], points[1, :], c=depth_points, s=30, vmin=vmin, vmax=vmax, cmap="jet")
    fig.colorbar(im, ax=ax, cmap="jet", pad=0.01)
    plt.show()


# Plot valid labels
def plot_valid_labels(image, points, valid_labels, vmin=0., vmax=2.):
    # Plot depth
    fig = plt.figure(figsize=(10,6))
    ax = plt.gca()
    plt.imshow(image)
    im = plt.scatter(points[0, :], points[1, :], c=valid_labels, s=30, vmin=vmin, vmax=vmax, cmap="jet")
    fig.colorbar(im, ax=ax, cmap="jet", pad=0.01)
    plt.show()


if __name__ == "__main__":
    # Construct the dataset object
    nuscene_dataset = Nuscenes_dataset()

    # Get samples
    sample_obj = nuscene_dataset.samples[90]

    orientation = "front"
    num_sweeps = 1

    # Get LiDAR and RADAR points separately
    lidar_data = nuscene_dataset.get_lidar_depth_map_multi_bidirectional(sample_obj, orientation, 1)
    radar_data = nuscene_dataset.get_radar_depth_map_multi_bidirectional(sample_obj, orientation, 3)

    filtered_data = filter_radar_points_gt(radar_data['points'],
                                        radar_data['depth_points'],
                                        lidar_data['points'],
                                        lidar_data['depth_points'])

    plot_lidar_depth(lidar_data['image'],
                     lidar_data['points'],
                     lidar_data['depth_points'], 0, 100)
    plot_radar_depth(radar_data['image'],
                     filtered_data['radar_points'],
                     filtered_data['radar_depth'], 0, 100)
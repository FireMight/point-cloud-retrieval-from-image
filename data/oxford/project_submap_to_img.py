#!/usr/bin/env python3
import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from ned_pcl_to_file import (import_trajectory_ned, get_pcl_metadata, 
                             plot_pcl_traj, get_pcl_segment)
from image_preprocessing import import_camera_trajectory, load_image
from pcl_trafo import pcl_trafo

# Import the point cloud builder from SDK
sys.path.insert(0, os.path.join(os.getcwd(),'robotcar-dataset-sdk/python'))
import build_pointcloud as sdk_pcl
from transform import build_se3_transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str)
    parser.add_argument('length', type=int)
    parser.add_argument('index', type=int)
    parser.add_argument('offset', type=float)
    parser.add_argument('camera', type=str)
    args = parser.parse_args()
    
    ins_data_file = 'data/2014-12-02-15-30-08/gps/ins.csv'
    vo_data_file = 'data/2014-12-02-15-30-08/vo/vo.csv'
    lidar_dir = 'data/2014-12-02-15-30-08/lms_front'
    lidar_timestamp_file = 'data/2014-12-02-15-30-08/lms_front.timestamps'
    extrinsics_dir = 'robotcar-dataset-sdk/extrinsics'
    
    if args.camera == 'left':
        camera_dir = 'data/2014-12-02-15-30-08/mono_left'
        camera_timestamp_file = 'data/2014-12-02-15-30-08/mono_left.timestamps'
        camera_extrinsics = 'robotcar-dataset-sdk/extrinsics/mono_left.txt'
    elif args.camera == 'right':
        camera_dir = 'data/2014-12-02-15-30-08/mono_right'
        camera_timestamp_file = 'data/2014-12-02-15-30-08/mono_right.timestamps'
        camera_extrinsics = 'robotcar-dataset-sdk/extrinsics/mono_right.txt'
    elif args.camera == 'center':
        camera_dir = 'data/2014-12-02-15-30-08/stereo/centre'
        camera_timestamp_file = 'data/2014-12-02-15-30-08/stereo.timestamps'
        camera_extrinsics = 'robotcar-dataset-sdk/extrinsics/stereo.txt'
    else:
        raise ValueError('Wrong camera type', args.camera)
    
    
    
    # Get trajectory and submap metadata
    trajectory_ned = import_trajectory_ned(ins_data_file, lidar_timestamp_file)
    pcl_dir = 'pcl/{}_{}m'.format(args.type, args.length)
    metadata_filename = pcl_dir + '/metadata_2014-12-02_{}_{}m.csv'.format(
                    args.type, args.length)
    pcl_metadata = get_pcl_metadata(metadata_filename, seg_idx=args.index)
    assert pcl_metadata is not None
    
    # Get start and end time for VO trajectory generation (begin 30m from submap 
    # start, end where submap ends)
    i_start = np.argwhere(trajectory_ned[6,:] == pcl_metadata['timestamp_start'])[0,0]
    dist_to_vo_start = 30
    prev_pos = trajectory_ned[:3,i_start]
    for i_vo_start in range(i_start-1,0,-1):
        curr_pos = trajectory_ned[[0,1,2],i_vo_start]
        dist_to_vo_start -= np.linalg.norm(prev_pos - curr_pos)
        prev_pos = curr_pos
        if dist_to_vo_start < 0:
            break
    
    dist_to_vo_end = args.length
    prev_pos = trajectory_ned[:3,i_start]
    for i_vo_end in range(i_start+1,trajectory_ned.shape[1]):
        curr_pos = trajectory_ned[[0,1,2],i_vo_end]
        dist_to_vo_end -= np.linalg.norm(prev_pos - curr_pos)
        prev_pos = curr_pos
        if dist_to_vo_end < 0:
            break
        
    # Get timestamp of image considering offset
    i_img = i_start
    if args.camera == 'center' and args.offset > 0:
        dist_to_img = args.offset
        prev_pos = trajectory_ned[:3,i_start]
        for i_img in range(i_start-1,0,-1):
            curr_pos = trajectory_ned[[0,1,2],i_img]
            dist_to_img -= np.linalg.norm(prev_pos - curr_pos)
            prev_pos = curr_pos
            if dist_to_img < 0:
                break
    elif args.camera in ['left', 'right']:
        dist_to_img = args.length / 2
        prev_pos = trajectory_ned[:3,i_start]
        for i_img in range(i_start+1,trajectory_ned.shape[1]):
            curr_pos = trajectory_ned[[0,1,2],i_img]
            dist_to_img -= np.linalg.norm(prev_pos - curr_pos)
            prev_pos = curr_pos
            if dist_to_img < 0:
                break
        
        
    
    # Build pointcloud based on visual odometry
    timestamp_start = trajectory_ned[6,i_vo_start]
    timestamp_img = trajectory_ned[6,i_img]
    timestamp_end = trajectory_ned[6,i_vo_end]
    print('Build PCL from VO with {} measurements...'.format(i_vo_end-i_vo_start))
    pointcloud, refl = sdk_pcl.build_pointcloud(lidar_dir, vo_data_file, 
                                             extrinsics_dir, timestamp_start, 
                                             timestamp_end, timestamp_img)
    print('Done!')

    # Transform pointcloud to NED and generate submap
    reference_state = trajectory_ned[:,i_img].flatten()
    pointcloud_ned = pcl_trafo(pointcloud, trans_newref=reference_state[:3], 
                               rot=reference_state[3:])
    
    i_center = np.argwhere(trajectory_ned[6,:] == pcl_metadata['timestamp_center'])[0,0]
    submap_ned, _ = get_pcl_segment(trajectory_ned, pointcloud_ned, i_center, 
                                    float(args.length), alignment='trajectory', 
                                    width=40)
    
    submap = pcl_trafo(submap_ned, trans_oldref=-reference_state[:3], 
                       rot=-reference_state[3:])
        
    # Get camera transformation
    with open(camera_extrinsics) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)
    submap = np.dot(G_camera_vehicle, submap)
    
    
    # Load image
    img_data = import_camera_trajectory(camera_timestamp_file, ins_data_file, 
                                        camera_dir)
    camera_trajectory, camera_model = img_data[0], img_data[1]
    camera_state = camera_trajectory[:,i_start]
    image, pil_image = load_image(camera_dir, camera_state[6], camera_model)
    
    # Project points into image
    uv, depth = camera_model.project(submap, image.shape)
    
    # Get ratio of points in image to total points
    points_in_image_ratio = uv.shape[1] / submap.shape[1]
    print('=========================')
    print('points in image / total points = {}'.format(points_in_image_ratio))

    plt.imshow(pil_image)
    #plt.hold(True)
    plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, 
                edgecolors='none', cmap='jet')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
    
    
    
    
    
        
        
    
    
    
    
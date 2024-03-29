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

ins_data_file = 'data/reference/gps/ins.csv'
vo_data_file = 'data/reference/vo/vo.csv'
lidar_dir = 'data/reference/lms_front'
lidar_timestamp_file = 'data/reference/lms_front.timestamps'
extrinsics_dir = 'robotcar-dataset-sdk/extrinsics'


def load_processed_submap(args):
    pcl_dir = 'data/reference/submaps_{}m'.format(args.length)
    submap_filename = pcl_dir + '/submap_{}.rawpcl.processed'.format(args.index)
    metadata_filename = pcl_dir + '/metadata.csv'
    
    # Load data 
    submap_ned = np.fromfile(submap_filename, dtype='float32')
    trajectory_ned = import_trajectory_ned(ins_data_file, lidar_timestamp_file)
    pcl_metadata = get_pcl_metadata(metadata_filename, seg_idx=args.index)
    assert pcl_metadata is not None
    
    # Rescale pointcloud
    submap_ned = submap_ned.reshape(3, submap_ned.shape[0]//3)
    center_pos = np.array([pcl_metadata['northing_center'],
                           pcl_metadata['easting_center'],
                           pcl_metadata['down_center']])
    center_pos = center_pos[:, np.newaxis]
    submap_ned = submap_ned.astype('float64') + center_pos
    
    # Add row of ones to pcl coordinates for trafo
    submap_ned = np.vstack((submap_ned, np.ones((1, submap_ned.shape[1]))))
    
    
    # Get image position
    i_start = np.argwhere(trajectory_ned[6,:] == pcl_metadata['timestamp_start'])[0,0]
    
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
    
    # Transform submap to vehicle reference system
    reference_state = trajectory_ned[:,i_img].flatten()
    submap = pcl_trafo(submap_ned, trans_oldref=-reference_state[:3], 
                       rot=-reference_state[3:])
    
    
    return i_img, submap

def generate_vo_submap(args):
    # Get trajectory and submap metadata
    trajectory_ned = import_trajectory_ned(ins_data_file, lidar_timestamp_file)
    pcl_dir = 'data/reference/submaps_{}m'.format(args.length)
    metadata_filename = pcl_dir + '/metadata.csv'
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
    
    print(submap)
    
    return i_img, submap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('length', type=int)
    parser.add_argument('index', type=int)
    parser.add_argument('offset', type=float)
    parser.add_argument('camera', type=str)
    args = parser.parse_args()
    
    use_processed = False
    
    
    if args.camera == 'left':
        camera_dir = 'data/reference/mono_left'
        camera_timestamp_file = 'data/reference/mono_left.timestamps'
        camera_extrinsics = 'robotcar-dataset-sdk/extrinsics/mono_left.txt'
    elif args.camera == 'right':
        camera_dir = 'data/reference/mono_right'
        camera_timestamp_file = 'data/reference/mono_right.timestamps'
        camera_extrinsics = 'robotcar-dataset-sdk/extrinsics/mono_right.txt'
    elif args.camera == 'center':
        camera_dir = 'data/reference/stereo/centre'
        camera_timestamp_file = 'data/reference/stereo.timestamps'
        camera_extrinsics = 'robotcar-dataset-sdk/extrinsics/stereo.txt'
    else:
        raise ValueError('Wrong camera type', args.camera)
    
    
    if use_processed:
        i_img, submap = load_processed_submap(args)
    else:
        i_img, submap = generate_vo_submap(args) 
    
    
            
    # Get camera transformation
    with open(camera_extrinsics) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)
    submap = np.dot(G_camera_vehicle, submap)
    
    
    # Load image
    img_data = import_camera_trajectory(camera_timestamp_file, ins_data_file, 
                                        camera_dir)
    camera_trajectory, camera_model = img_data[0], img_data[1]
    camera_state = camera_trajectory[:,i_img] #Was i_start, not correct?
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
    
    
    
    
    
    
        
        
    
    
    
    
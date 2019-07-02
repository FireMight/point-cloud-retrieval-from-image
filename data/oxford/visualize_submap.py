#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

from ned_pcl_to_file import import_trajectory_ned, get_pcl_metadata, plot_pcl_traj



if __name__ == "__main__":
    ins_data_file = 'data/2014-12-02-15-30-08/gps/ins.csv'
    lidar_timestamp_file = 'data/2014-12-02-15-30-08/lms_front.timestamps'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str)
    parser.add_argument('length', type=int)
    parser.add_argument('index', type=int)
    args = parser.parse_args()
    
    pcl_dir = 'pcl/{}_{}m'.format(args.type, args.length)
    submap_filename = pcl_dir + '/submap_2014-12-02_{}_{}m_{}.rawpcl'.format(
                    args.type, args.length, args.index)
    metadata_filename = pcl_dir + '/metadata_2014-12-02_{}_{}m.csv'.format(
                    args.type, args.length)
    
    # Load data 
    submap = np.fromfile(submap_filename, dtype='float32')
    trajectory_ned = import_trajectory_ned(ins_data_file, lidar_timestamp_file)
    metadata = get_pcl_metadata(metadata_filename, seg_idx=args.index)
    assert metadata is not None
    
    # Rescale pointcloud
    submap = submap.reshape(3, submap.shape[0]//3)
    center_pos = np.array([metadata['northing_center'],
                           metadata['easting_center'],
                           metadata['down_center']])
    center_pos = center_pos[:, np.newaxis]
    submap = submap.astype('float64') + center_pos
    
        
    # Plot on map
    plt.scatter(trajectory_ned[1,:], trajectory_ned[0,:], c='gray', s=0.5, 
                marker='.')
    plt.scatter(float(metadata['easting_start']), float(metadata['northing_start']), 
                c='green', s=20, marker='o')
    plt.scatter(float(metadata['easting_center']), float(metadata['northing_center']), 
                c='red', s=20, marker='o')
    plt.show()
    
    # Plot pointcloud
    plot_pcl_traj(submap, trajectory_ned=trajectory_ned)
                
    
    
        
    
#!/usr/bin/env python3
from ned_pcl_to_file import *


if __name__ == "__main__":
    # Replace with arg parser later if required
    ins_data_file = 'data_sample/gps/ins.csv'
    lidar_dir = 'data_sample/lms_front'
    lidar_timestamp_file = 'data_sample/lms_front.timestamps'
    
    submap_coverages = [10, 20, 30]
    img_offset = [0, 10]
    
    data_ned = import_trajectory_pcl_ned(lidar_timestamp_file, lidar_dir, 
                                         ins_data_file)
    trajectory_ned, pointcloud_ned, reflectance = (data_ned[0], data_ned[1], 
                                                   data_ned[2])
    
    # Pick "random sample" (roughly in the middle of the segment for testing)
    sample_idx = 400
    
    submap, reflectance_seg = get_pcl_segment(trajectory_ned, pointcloud_ned, 
                                              sample_idx, 20, alignment='trajectory',
                                              width=50, reflectance=reflectance)
    
    plot_pcl_traj(submap, reflectance=reflectance_seg)
    
    
    
    
#!/usr/bin/env python3

import csv
import numpy as np
import os
import sys
from datetime import datetime
from collections import namedtuple, defaultdict

from pcl_trafo import pcl_trafo

# Import the point cloud builder from SDK
sys.path.insert(0, os.path.join(os.getcwd(),'robotcar-dataset-sdk/python'))
import build_pointcloud as sdk_pcl


def import_trajectory_ned(ins_data_file, lidar_timestamp_file):
    # Get start and end timestamp of LIDAR measurements
    print('Get start and end time of LIADR measurements...')
    with open(lidar_timestamp_file) as ts_file:
        start_time = int(next(ts_file).split(' ')[0])
        for end_time in ts_file:
            continue
        end_time = int(end_time.split(' ')[0])
    print('Done!')
    
    # Check if NED trajectory has already been computed
    date_of_run = datetime.utcfromtimestamp(start_time*1e-6).strftime('%Y-%m-%d')
    trajectory_file = 'trajectory/ned_trajectory_{}.npy'.format(date_of_run)
    print('Search for precomputed NED trajectory ' + trajectory_file + '...')
    
    try:
        trajectory_ned = np.load(trajectory_file)
        print('Done!')
    except:
        # Get trajectory corresponding to LIDAR data
        print('Not found! Create NED trajectory from INS data...')
        trajectory_ned = np.empty((7,0))
        with open(ins_data_file, 'r') as ins_file:
            reader = csv.DictReader(ins_file)
            for row in reader:
                if int(row['timestamp']) > end_time:
                    break
                if int(row['timestamp']) < start_time:
                    continue
                ned_state = np.array([float(row['northing']),
                                      float(row['easting']),
                                      float(row['down']),
                                      float(row['roll']),
                                      float(row['pitch']),
                                      float(row['yaw']),
                                      float(row['timestamp'])]).reshape(7,1)
                trajectory_ned = np.append(trajectory_ned, ned_state, axis=1)
        print('Done! Trajectory with {} samples'.format(trajectory_ned.shape[1]))
        
        # Save trajectory
        print('Save NED trajectory to ' + trajectory_file + '...')
        np.save(trajectory_file, trajectory_ned)
        print('Done!')
    
    return trajectory_ned


def get_next_split(i_start, max_trajectory_size, trajectory_ned, lidar_dir, 
                   extrinsics_dir='robotcar-dataset-sdk/extrinsics'):
    split = namedtuple('Split', 'i_end, end_of_route, pcl_ned, reflectance')
    state_first_frame = trajectory_ned[:,i_start]
    i_max = min(i_start+max_trajectory_size, trajectory_ned.shape[1])
    
    prev_pos = state_first_frame[:3]
    dist_to_20 = 20
    split.i_end = None
    i_pcl_end = None
    n_submaps_20 = 0
    for i in range(i_start+1,i_max):
        curr_pos = trajectory_ned[[0,1,2],i]
        dist_to_20 -= np.linalg.norm(prev_pos - curr_pos)
        prev_pos = curr_pos
        
        if dist_to_20 < 0:
            if i_pcl_end is not None:
                split.i_end = i_pcl_end
                n_submaps_20 += 1
            
            i_pcl_end = i
            dist_to_20 = 20

    print('Get next split: i_start {} i_end {} i_pcl_end {} n_submaps_20 {}'.
          format(i_start, split.i_end, i_pcl_end, n_submaps_20))
            
    if split.i_end is None or i_pcl_end is None:
        split.end_of_route = True
    else:
        split.end_of_route = False
        start_time = state_first_frame[6]
        end_time = trajectory_ned[6,i_pcl_end]
        
        # Build pointcloud in vehicle reference frame
        print('Build PCL...')
        pcl_split_veh, split.reflectance = sdk_pcl.build_pointcloud(lidar_dir, 
                                            ins_data_file, extrinsics_dir, 
                                            start_time, end_time)
        print('Done!')
        
        # Transform pointlcoud to NED system
        state_first_frame = state_first_frame.flatten()
        split.pcl_ned = pcl_trafo(pcl_split_veh, trans_newref=state_first_frame[:3], 
                                  rot=state_first_frame[3:])

    return split


def create_ref_submaps(i_start, i_end, length, pcl_ned, trajectory_ned, 
                       date_of_run, metadata_csv, segment_idx_start):
    print('Create reference submaps for length {} (i_start {}, i_end {})...'.
          format(length, i_start, i_end))
    directory = 'pcl/reference_{}m'.format(length)
    prev_pos = trajectory_ned[[0,1,2],i_start]
    dist_to_next = length
    i_center = None
    segment_idx = segment_idx_start
    
    for i in range(i_start+1,i_end+2): # +2 to account for rounding errors
        curr_pos = trajectory_ned[[0,1,2],i]
        dist_to_next -= np.linalg.norm(prev_pos - curr_pos)
        prev_pos = curr_pos
                
        # Save center point when half of the distance is travelled
        if dist_to_next<=length/2 and i_center is None:
            i_center = i
        
        # Split trajectory when full distance is travelled
        if dist_to_next<=0:
            submap, _ = get_pcl_segment(trajectory_ned, pcl_ned, 
                                        i_center, length, width=50,
                                        alignment='trajectory')
            submap = submap[:3,:]
            
            # Save raw pointcloud for preprocessing with PCL later
            submap.tofile(directory + '/submap_{}_reference_{}m_{}.rawpcl'.
                          format(date_of_run, length, segment_idx))
    
            #Save metadata of all pointcloud files to csv
            with open(metadata_csv, 'a') as outcsv:
                writer = csv.DictWriter(outcsv,fieldnames=metadata_fieldnames)
                writer.writerow({'seg_idx' : segment_idx, 
                                 'timestamp_start' : int(trajectory_ned[6,i_start]), 
                                 'northing_start' : trajectory_ned[0,i_start],
                                 'easting_start' : trajectory_ned[1,i_start],
                                 'down_start' : trajectory_ned[2,i_start],
                                 'heading_start' : trajectory_ned[5,i_start],
                                 'timestamp_center' : int(trajectory_ned[6,i_center]),
                                 'northing_center' : trajectory_ned[0,i_center],
                                 'easting_center' : trajectory_ned[1,i_center],
                                 'down_center' : trajectory_ned[2,i_center],
                                 'heading_center' : trajectory_ned[5,i_center]})
    
            i_start = i+1
            dist_to_next=length
            i_center = None
            segment_idx += 1
    
    print('Done! Created {} submaps'.format(segment_idx-segment_idx_start))
    
    return segment_idx


def create_rand_submaps(i_min, i_max, length, pcl_ned, trajectory_ned, 
                        n_submaps, date_of_run, metadata_csv, segment_idx_start):
    print('Create {} random submaps for length {} (i_min {}, i_max {})...'.
          format(n_submaps, length, i_min, i_max))
    directory = 'pcl/random_{}m'.format(length)
    segment_idx = segment_idx_start
    
    for _ in range(n_submaps):
        # Get random start index and corresponding center 
        i_start = np.random.randint(low=i_min, high=i_max)
        prev_pos = trajectory_ned[[0,1,2],i_start]
        dist_to_center = length/2
        
        for i_center in range(i_start, i_max):
            curr_pos = trajectory_ned[[0,1,2],i_center]
            dist_to_center -= np.linalg.norm(prev_pos - curr_pos)
            prev_pos = curr_pos
            
            if dist_to_center < 0:
                break
        
        # Create submap
        submap, _ = get_pcl_segment(trajectory_ned, pcl_ned, 
                                    i_center, length, width=50,
                                    alignment='trajectory')
        submap = submap[:3,:]
        
        # Save raw pointcloud for preprocessing with PCL later
        submap.tofile(directory + '/submap_{}_random_{}m_{}.rawpcl'.
                      format(date_of_run, length, segment_idx))

        #Save metadata of all pointcloud files to csv
        with open(metadata_csv, 'a') as outcsv:
            writer = csv.DictWriter(outcsv,fieldnames=metadata_fieldnames)
            writer.writerow({'seg_idx' : segment_idx, 
                             'timestamp_start' : int(trajectory_ned[6,i_start]), 
                             'northing_start' : trajectory_ned[0,i_start],
                             'easting_start' : trajectory_ned[1,i_start],
                             'down_start' : trajectory_ned[2,i_start],
                             'heading_start' : trajectory_ned[5,i_start],
                             'timestamp_center' : int(trajectory_ned[6,i_center]),
                             'northing_center' : trajectory_ned[0,i_center],
                             'easting_center' : trajectory_ned[1,i_center],
                             'down_center' : trajectory_ned[2,i_center],
                             'heading_center' : trajectory_ned[5,i_center]})
        
        segment_idx += 1
    print('Done! Created {} submaps'.format(segment_idx-segment_idx_start))
    
    return segment_idx
        

def get_pcl_segment(trajectory_ned, pointcloud_ned, center_idx, coverage, 
                    alignment='north_east', width=None, reflectance=None):
    # Get center position on trajectory
    center_pos = trajectory_ned[[0,1,2], center_idx]  
    reflectance_segment = None
    
    if alignment == 'north_east':
        # Get all points with north and east coordinate within coverage/2
        box_min = center_pos - coverage/2
        box_max = center_pos + coverage/2
        mask = np.array(np.logical_and(np.logical_and(pointcloud_ned[0,:]>=box_min[0],pointcloud_ned[0,:]<box_max[0]),
            np.logical_and(pointcloud_ned[1,:]>=box_min[1], pointcloud_ned[1,:]<box_max[1]))).squeeze()
        pcl_segment = pointcloud_ned[:,mask]
        
        if reflectance is not None:
            reflectance_segment = reflectance[mask]
    
    elif alignment == 'trajectory':        
        # Bounding box length in trajectory direction, optional width orthogonal
        center_heading = trajectory_ned[5, center_idx]
        
        # Consider different width if specified, else use quadratic box
        if width is None:
            width = coverage
        
        # Only considere points within certain range of the center point
        r_max = np.sqrt(2 * pow(max(coverage, width)/2, 2))
        r = np.linalg.norm(pointcloud_ned[:2,:] - center_pos[:2].reshape(2,1), 
                           axis=0)
        
        pcl_ned = pointcloud_ned[:, r < r_max]
        
        # Rotate pointcloud into bounding box reference system.
        pcl_bb = pcl_trafo(pcl_ned, trans_oldref=-center_pos, 
                           rot=np.array([0, 0, -center_heading]))
        
        # Get extend of bounding box
        box_max = np.array([coverage/2, width/2, 0])
        box_min = -1 * box_max
        
        mask = np.array(np.logical_and(np.logical_and(pcl_bb[0,:]>=box_min[0],pcl_bb[0,:]<box_max[0]),
            np.logical_and(pcl_bb[1,:]>=box_min[1], pcl_bb[1,:]<box_max[1]))).squeeze()

        # Get segment from untransformed PCL
        pcl_segment = pcl_ned[:,mask]
        
        if reflectance is not None:
            reflectance_limited = reflectance[r < r_max]
            reflectance_segment = reflectance_limited[mask]
    
    else:
        raise ValueError('Wrong bounding box alignment specified: ' + alignment)
    
    
    return pcl_segment, reflectance_segment
    

if __name__ == "__main__":
    # Replace with arg parser later if required
    ins_data_file = 'data/2014-12-02-15-30-08/gps/ins.csv'
    lidar_dir = 'data/2014-12-02-15-30-08/lms_front'
    lidar_timestamp_file = 'data/2014-12-02-15-30-08/lms_front.timestamps'
    max_trajectory_size = 2000 # per split
    n_random_submaps = 3000 # per length
    lengths = [10,20] # IMPORTANT: Works only for these values, do not change!
    
    # Load NED trajectory
    trajectory_ned = import_trajectory_ned(ins_data_file, lidar_timestamp_file)
    n_meas = trajectory_ned.shape[1]
    date_of_run = datetime.utcfromtimestamp(
            trajectory_ned[6,0]*1e-6).strftime('%Y-%m-%d')
            
    # Create csv metadata files
    metadata_fieldnames = ['seg_idx', 
                           'timestamp_start',
                           'northing_start',
                           'easting_start',
                           'down_start',
                           'heading_start', 
                           'timestamp_center',
                           'northing_center',
                           'easting_center',
                           'down_center',
                           'heading_center']
    metadata_csv = {}
    for label in ['reference', 'random']:
        metadata_csv[label] = {}
        for length in lengths:
            directory = 'pcl/{}_{}m'.format(label, length)
            metadata_csv[label][length] = (directory + 
                                           '/metadata_{}_{}_{}m.csv'.
                                           format(date_of_run, label, length))
                            
            with open(metadata_csv[label][length], 'w') as outcsv:
                writer = csv.DictWriter(outcsv, metadata_fieldnames)
                writer.writeheader()
    
    # Calculate required number of random submaps per split
    n_splits = trajectory_ned.shape[1] / max_trajectory_size
    n_random_split = int(n_random_submaps / n_splits)
    
    
    # Initialize start timestamps and segment indices
    i_start = 0
    seg_idx_start = defaultdict(lambda: defaultdict(int))
    while True:
        print('============= new split =============')
        split = get_next_split(i_start, max_trajectory_size, trajectory_ned, 
                               lidar_dir)
        
        if split.end_of_route:
            print('End of route')
            break
        
        for length in lengths:
            # Create equally spaced reference submaps for testing
            seg_idx_start['reference'][length] = create_ref_submaps(i_start, 
                                    split.i_end, length, split.pcl_ned, 
                                    trajectory_ned, date_of_run,
                                    metadata_csv['reference'][length], 
                                    seg_idx_start['reference'][length])
            
            # Create random submaps for training
            seg_idx_start['random'][length] = create_rand_submaps(i_start, 
                                    split.i_end, length, split.pcl_ned, 
                                    trajectory_ned, n_random_split, date_of_run,
                                    metadata_csv['random'][length], 
                                    seg_idx_start['random'][length])
            
        i_start = split.i_end + 1

        
            
                


        

    
    
#!/usr/bin/env python3

import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys
from random import sample
from datetime import datetime

from pcl_trafo import pcl_trafo

# Import the point cloud builder from SDK
sys.path.insert(0, os.path.join(os.getcwd(),'robotcar-dataset-sdk/python'))
import build_pointcloud as sdk_pcl


def plot_pcl_traj(pointcloud_ned, reflectance=None, trajectory_ned=None):
    x = np.ravel(pointcloud_ned[0, :])
    y = np.ravel(pointcloud_ned[1, :])
    z = np.ravel(pointcloud_ned[2, :])

    xmin = x.min()
    ymin = y.min()
    zmin = z.min()
    xmax = x.max()
    ymax = y.max()
    zmax = z.max()
    xmid = (xmax + xmin) * 0.5
    ymid = (ymax + ymin) * 0.5
    zmid = (zmax + zmin) * 0.5

    max_range = max(xmax - xmin, ymax - ymin, zmax - zmin)
    x_range = [xmid - 0.5 * max_range, xmid + 0.5 * max_range]
    y_range = [ymid - 0.5 * max_range, ymid + 0.5 * max_range]
    z_range = [zmid - 0.5 * max_range, zmid + 0.5 * max_range]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')
    
    if reflectance is not None:
        colours = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
        colours = 1 / (1 + np.exp(-10 * (colours - colours.mean())))
        ax.scatter(-y, -x, -z, marker=',', s=1, c=colours, cmap='gray', 
               edgecolors='none')
    else:
        ax.scatter(-y, -x, -z, marker=',', s=1, c='gray',
               edgecolors='none')
    
    if trajectory_ned is not None:
        traj_x = np.ravel(trajectory_ned[0, :])
        traj_y = np.ravel(trajectory_ned[1, :])
        traj_z = np.ravel(trajectory_ned[2, :])
        ax.scatter(-traj_y, -traj_x, -traj_z, marker=',', s=1, c='r')
        
    ax.set_xlim(-y_range[1], -y_range[0])
    ax.set_ylim(-x_range[1], -x_range[0])
    ax.set_zlim(-z_range[1], -z_range[0])
    ax.view_init(140, 0) # elevation, azimuth
    plt.show()
    

def import_trajectory_pcl_ned(lidar_timestamp_file, lidar_dir, ins_data_file, 
                              extrinsics_dir='robotcar-dataset-sdk/extrinsics'):
    # Get start and end timestamp
    max_frames = 1e9
    with open(lidar_timestamp_file) as ts_file:
        start_time = int(next(ts_file).split(' ')[0])
        for it, end_time in enumerate(ts_file):
            if it > max_frames:
                break
            continue
        end_time = int(end_time.split(' ')[0])
    
    # Calculate point cloud relative to start frame of trajectory
    pcl_veh, reflectance = sdk_pcl.build_pointcloud(lidar_dir, ins_data_file,  
                                                    extrinsics_dir, start_time, 
                                                    end_time)
    
    # Get trajectory corresponding to LIDAR data
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
                                  int(row['timestamp'])]).reshape(7,1)
            trajectory_ned = np.append(trajectory_ned, ned_state, axis=1)
    
    # Create transformation matrix and transform pcl from vehicle-fixed to NED
    state_first_frame = trajectory_ned[:6,0]
    state_first_frame = state_first_frame.flatten()
    pcl_ned = pcl_trafo(pcl_veh, trans_newref=state_first_frame[:3], 
                        rot=state_first_frame[3:])
    
    return trajectory_ned, pcl_ned, reflectance


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


def simple_pcl_downsampling(pcl, target_size, ground_frac):
    #TODO: rudimentary ground removal. replace with SAC removal later
    down_coord = np.array(pcl[2,:]).squeeze()
    rng = max(down_coord) - min(down_coord)
    ground_thresh = max(down_coord)-rng*ground_frac
    pcl_downsample = pcl[:,down_coord<ground_thresh]
    
    if pcl_downsample.shape[1] >= 3*target_size:
        subsample = sample(list(range(pcl_downsample.shape[1])),target_size)
        pcl_downsample = pcl_downsample[:,subsample]
    else:
        pcl_downsample = None
    
    return pcl_downsample
    

if __name__ == "__main__":
    # Replace with arg parser later if required
    ins_data_file = 'data_sample/gps/ins.csv'
    lidar_dir = 'data_sample/lms_front'
    lidar_timestamp_file = 'data_sample/lms_front.timestamps'
    plot = True
    to_file = True
    #what fraction of the down axis is considered ground
    ground_frac = 0.1
    subsample_size = 4096
    submap_coverage = 25.0
    
    data_ned = import_trajectory_pcl_ned(lidar_timestamp_file, lidar_dir, 
                                         ins_data_file)
    trajectory_ned, pointcloud_ned, reflectance = (data_ned[0], data_ned[1], 
                                                   data_ned[2])
    
    
    if plot:
        plot_pcl_traj(pointcloud_ned,reflectance,trajectory_ned)
    
    if to_file:
        dist_to_next = submap_coverage
        prev_pos = trajectory_ned[[0,1,2],0]
        segment_center_idx = None
        i_start = 0 # Index of the start of each segment
        segment_idx = 0
        date_of_run = datetime.utcfromtimestamp(
            trajectory_ned[6,0]*1e-6).strftime('%Y-%m-%d')
        metadata_csv = 'pcl/segments_metadata_{}.csv'.format(date_of_run)
        metadata_fieldnames = ['seg_idx', 
                               'timestamp_start', 
                               'timestamp_end',
                               'northing_start',
                               'easting_start',
                               'down_start',
                               'northing_end',
                               'easting_end',
                               'down_end',
                               'heading_start',
                               'heading_end']
            
        # Create csv file containing metadata for all segments
        with open(metadata_csv, 'w') as outcsv:
            writer = csv.DictWriter(outcsv, metadata_fieldnames)
            writer.writeheader()
        
        #TODO: check units - are point cloud coordinates given in meters
        #walk along trajectory, sample every submap_coverage units traveled
        for i in range(1,trajectory_ned.shape[1]):
            curr_pos = trajectory_ned[[0,1,2],i]
            dist_to_next = dist_to_next - np.linalg.norm(prev_pos - curr_pos)
            prev_pos=curr_pos
            
            # Save center point when half of the distance is travelled
            if dist_to_next<=submap_coverage/2 and segment_center_idx is None:
                segment_center_idx = i
            
            # Split trajectory when full distance is travelled
            if dist_to_next<=0:
                submap, _ = get_pcl_segment(trajectory_ned, pointcloud_ned, 
                                            segment_center_idx, submap_coverage)
                submap = submap[:3,:]
                
                # Save raw pointcloud for preprocessing with PCL later
                submap.tofile('pcl/oxford_{}_{}.rawpcl'.format(date_of_run, segment_idx))
                
                # Rudimentary ground removal and downsampling
                submap = simple_pcl_downsampling(submap, subsample_size, ground_frac)
                
                #if submap doesn't contain enough points, skip to next submap;
                #this shouldn't happen, warn if it does
                if submap is None:
                    print("No submap for pos {} generated, point cloud density too low, idx ", 
                          segment_center_idx)
                    continue
                
                #center and rescale to the range [-1,1]
                segment_center_pos = trajectory_ned[[0,1,2], segment_center_idx]
                submap[0,:]=submap[0,:]-segment_center_pos[0]
                submap[1,:]=submap[1,:]-segment_center_pos[1]
                submap[2,:]=submap[2,:]-segment_center_pos[2]
                submap = submap/(submap_coverage/2)

                submap.tofile('pcl/oxford_{}_{}.pcl'.format(date_of_run, segment_idx))
 
                #Save metadata of all pointcloud files to csv
                with open(metadata_csv, 'a') as outcsv:
                    writer = csv.DictWriter(outcsv,fieldnames=metadata_fieldnames)
                    writer.writerow({'seg_idx' : segment_idx, 
                                     'timestamp_start' : int(trajectory_ned[6,i_start]), 
                                     'timestamp_end' : int(trajectory_ned[6,i]),
                                     'northing_start' : trajectory_ned[0,i_start],
                                     'easting_start' : trajectory_ned[1,i_start],
                                     'down_start' : trajectory_ned[2,i_start],
                                     'northing_end' : trajectory_ned[0,i],
                                     'easting_end' : trajectory_ned[1,i],
                                     'down_end' : trajectory_ned[2,i],
                                     'heading_start' : trajectory_ned[5,i_start],
                                     'heading_end' : trajectory_ned[5,i]})
                
                if plot:
                    trajectory_rescaled = trajectory_ned[:,i_start:i]
                    trajectory_rescaled[0,:] = trajectory_rescaled[0,:] - segment_center_pos[0]
                    trajectory_rescaled[1,:] = trajectory_rescaled[1,:] - segment_center_pos[1]
                    trajectory_rescaled[2,:] = trajectory_rescaled[2,:] - segment_center_pos[2]
                    trajectory_rescaled[0:3,:] = trajectory_rescaled[0:3,:] / (submap_coverage/2)
                    plot_pcl_traj(submap, trajectory_ned=trajectory_rescaled)

                i_start = i+1
                dist_to_next=submap_coverage
                segment_center_pos = None
                segment_idx += 1
    

    
    
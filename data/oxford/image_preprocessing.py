#!/usr/bin/env python3

import csv
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image

# Import image rectification and debayering from SDK
sys.path.insert(0, os.path.join(os.getcwd(),'robotcar-dataset-sdk/python'))
import image as sdk_image
from camera_model import CameraModel


def get_closest_camera_timestamp(ins_timestamp, camera_timestamps, start_idx):
    min_diff = 1e9
    i_min = start_idx
    for i in range(start_idx, camera_timestamps.shape[0]):
        diff = math.fabs(ins_timestamp - camera_timestamps[i])
        
        if diff < min_diff:
            min_diff = diff
            i_min = i
        else:
            break
        
    return camera_timestamps[i_min], i_min

def import_camera_trajectory(camera_timestamp_file, ins_data_file, camera_dir,
                             models_dir = 'robotcar-dataset-sdk/models'):
    # Import image rectification and debayering from SDK
    camera_model = CameraModel(models_dir, camera_dir)
    
    # Get timestamps from camera data
    camera_timestamps = np.genfromtxt(camera_timestamp_file)
    camera_timestamps = camera_timestamps[:,0].flatten()
    start_time = camera_timestamps[0]
    end_time = camera_timestamps[-1]
            
    # Check if camera trajectory has already been computed
    date_of_run = datetime.utcfromtimestamp(start_time*1e-6).strftime('%Y-%m-%d')
    trajectory_file = 'trajectory/ned_trajectory_{}_{}.npy'.format(camera_model.camera,
                                                                   date_of_run)
    print('Search for precomputed camera trajectory ' + trajectory_file + '...')
    
    try:
        camera_trajectory = np.load(trajectory_file)
        print('Done!')
    except:
        print('Not found! Create camera trajectory...')
        # Get trajectory corresponding to camera data
        camera_timestamp_idx = 0
        camera_trajectory = np.empty((7,0))
        with open(ins_data_file, 'r') as ins_file:
            reader = csv.DictReader(ins_file)
            for row in reader:
                if int(row['timestamp']) > end_time:
                    break
                if int(row['timestamp']) < start_time:
                    continue
                
                # Get closest corresponding camera timestamp
                timestamp, camera_timestamp_idx = get_closest_camera_timestamp(
                    int(row['timestamp']), camera_timestamps, camera_timestamp_idx)
                
                camera_state = np.array([float(row['northing']),
                                         float(row['easting']),
                                         float(row['down']),
                                         float(row['roll']),
                                         float(row['pitch']),
                                         float(row['yaw']),
                                         timestamp]).reshape(7,1)
                camera_trajectory = np.append(camera_trajectory, camera_state, axis=1)
        
        # Save trajectory
        print('Save camera trajectory to ' + trajectory_file + '...')
        np.save(trajectory_file, camera_trajectory)
        print('Done!')
    
    return camera_trajectory, camera_model

def load_image(camera_dir, timestamp, camera_model, size=None):
    image_file = os.path.join(camera_dir, '{}.png'.format(int(timestamp)))
    image = sdk_image.load_image(image_file, model=camera_model)
    
    pil_image = Image.fromarray(image.astype('uint8'))
    if size is not None:
        pil_image = pil_image.resize(size)
    
    return image, pil_image


if __name__ == "__main__":
    camera_dir = 'data_sample/stereo/centre'
    camera_timestamp_file = 'data_sample/stereo.timestamps'
    ins_data_file = 'data_sample/gps/ins.csv'
    ref_trajectory_file = 'pcl/segments_metadata_2014-12-12.csv'
    plot = False
    
    # Get camera timestamps and model
    camera_trajectory, camera_model = import_camera_trajectory(camera_timestamp_file, 
                                                    ins_data_file, camera_dir)
    
    start_time = camera_trajectory[6,0]
    end_time = camera_trajectory[6,-1]
    
    # Get trajectory corresponding to reference point cloud map
    ref_trajectory = np.empty((6,0))
    with open(ref_trajectory_file, 'r') as ref_file:
        reader = csv.DictReader(ref_file)
        for row in reader:
            if float(row['timestamp_end']) > end_time:
                break
            if float(row['timestamp_start']) < start_time:
                continue
            segment_start = np.array([float(row['northing_start']),
                                      float(row['easting_start']),
                                      float(row['down_start']),
                                      float(row['heading_start']),
                                      int(row['seg_idx']),
                                      int(row['timestamp_start'])]).reshape(6,1)
            ref_trajectory = np.append(ref_trajectory, segment_start, axis=1)
    
    
    # Create csv file containing metadata for all images
    date_of_run = datetime.utcfromtimestamp(
            camera_trajectory[6,0]*1e-6).strftime('%Y-%m-%d')
    metadata_csv = 'img/image_metadata_{}.csv'.format(date_of_run)
    metadata_fieldnames = ['img_idx',
                           'seg_idx', 
                           'timestamp', 
                           'northing',
                           'easting',
                           'down',
                           'heading',
                           'dist_to_seg_start']
            
    with open(metadata_csv, 'w') as outcsv:
        writer = csv.DictWriter(outcsv, metadata_fieldnames)
        writer.writeheader()

    
    # Save all images that are within a specified distance to the start point 
    # of a trajectory segment
    max_dist = 1.0
    max_head_diff = math.radians(10.0)
    img_idx = 0
    for camera_idx in range(camera_trajectory.shape[1]):
        camera_state = camera_trajectory[:,camera_idx]
        for seg_idx in range(ref_trajectory.shape[1]):
            seg_start = ref_trajectory[:,seg_idx]
            dist = np.linalg.norm(camera_state[0:3] - seg_start[0:3])
            head_diff = math.fabs(camera_state[5] - seg_start[3])
            
            if dist > max_dist or head_diff > max_head_diff:
                continue
            
            
            # Load image, save image and metadata
            _, pil_image = load_image(camera_dir, camera_state[6], camera_model,
                                      size=(320,240))
            pil_image.save('img/oxford_{}_{}_{}.png'.format(
                            date_of_run,int(seg_start[4]),img_idx), 'PNG')
            
            with open(metadata_csv, 'a') as outcsv:
                writer = csv.DictWriter(outcsv,fieldnames=metadata_fieldnames)
                writer.writerow({'img_idx' : img_idx,
                                 'seg_idx' : int(seg_start[4]), 
                                 'timestamp' : int(camera_state[6]), 
                                 'northing' : camera_state[0],
                                 'easting' : camera_state[1],
                                 'down' : camera_state[2],
                                 'heading' : camera_state[5],
                                 'dist_to_seg_start' : dist})
            
            img_idx += 1
                
            if plot:
                plt.imshow(pil_image)
                plt.xlabel('Segment {}'.format(seg_start[4]))
                plt.xticks([])
                plt.yticks([])
                plt.pause(0.05)
                
                
        
        
        
        
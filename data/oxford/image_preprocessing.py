#!/usr/bin/env python3

import csv
import os
import sys
import math
import numpy as np
from datetime import datetime
from PIL import Image

from ned_pcl_to_file import get_pcl_metadata

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


def save_images_to_pcl(data_dir, ins_data_file, label, length, camera, 
                       center_offset=0):
    # Load pointcloud metadata
    pcl_dir = 'pcl/{}_{}m'.format(label, length)
    metadata_filename = (pcl_dir + '/metadata_2014-12-02_{}_{}m.csv'.
                                   format(label, length))
    pcl_metadata = get_pcl_metadata(metadata_filename)
    
    # Sort list by timestamp
    pcl_metadata.sort(key=lambda pcl: pcl['timestamp_start'])
    
    # Load camera trajectory and model
    if camera == 'center':
        img_dir = data_dir + '/stereo/centre'
        timestamp_file = data_dir + '/stereo.timestamps'
    else:
        img_dir = data_dir + '/mono_{}'.format(camera)
        timestamp_file = data_dir + '/mono_{}.timestamps'.format(camera)
    
    camera_trajectory, camera_model = import_camera_trajectory(
                                        timestamp_file, ins_data_file, img_dir)
    date_of_run = datetime.utcfromtimestamp(
                               camera_trajectory[6,0]*1e-6).strftime('%Y-%m-%d')
    
    # Create image metadata csv file
    metadata_fieldnames = ['seg_idx',
                           'camera',
                           'offset', 
                           'timestamp',
                           'northing',
                           'easting',
                           'down',
                           'heading', 
                           'dist_to_ref']
    target_dir = 'img/{}_{}m'.format(label, length)
    img_metadata_csv = (target_dir + '/metadata_{}_{}_{}m.csv'.
                        format(date_of_run, label, length))
                    
    with open(img_metadata_csv, 'w') as outcsv:
        writer = csv.DictWriter(outcsv, metadata_fieldnames)
        writer.writeheader()
    
    
    # Walk along camera trajectory and save all images belonging to a submap
    if camera == 'center':
        ref = 'start'
        offset = center_offset
    else:
        ref = 'center'
        offset = 0
        
    i_pcl = 0
    ref_pos = np.array([pcl_metadata[i_pcl]['northing_'+ref],
                        pcl_metadata[i_pcl]['easting_'+ref]])
    dist_to_ref_prev = sys.maxsize
    for i_cam in range(camera_trajectory.shape[1]):
        pos = camera_trajectory[:2,i_cam]
        dist_to_ref = np.linalg.norm(pos - ref_pos) - offset
        
        if dist_to_ref > dist_to_ref_prev:
            # Save image of previous step as png
            timestamp = camera_trajectory[6,i_cam-1]
            seg_idx = pcl_metadata[i_pcl]['seg_idx']
            
            _, pil_image = load_image(img_dir, timestamp, camera_model)
            pil_image.save(target_dir + '/{}_{}m_{}_{}_{}m_{}.png'.format(
                           camera, int(center_offset), date_of_run, label, 
                           length, seg_idx), 'PNG')
            
            # Save image metadata
            with open(img_metadata_csv, 'a') as outcsv:
                writer = csv.DictWriter(outcsv,fieldnames=metadata_fieldnames)
                writer.writerow({'seg_idx' : seg_idx,
                                 'camera' : camera,
                                 'offset' : center_offset, 
                                 'timestamp' : timestamp,
                                 'northing' : camera_trajectory[0,i_cam-1],
                                 'easting' : camera_trajectory[1,i_cam-1],
                                 'down' : camera_trajectory[2,i_cam-1],
                                 'heading' : camera_trajectory[5,i_cam-1], 
                                 'dist_to_ref' : dist_to_ref_prev})
            

            # Set new reference position
            i_pcl += 1
            ref_pos = np.array([pcl_metadata[i_pcl]['northing_'+ref],
                        pcl_metadata[i_pcl]['easting_'+ref]])
            dist_to_ref_prev = sys.maxsize
            
        dist_to_ref_prev = dist_to_ref



if __name__ == "__main__":
    data_dir='data/2014-12-02-15-30-08'
    
    for label in ['reference', 'random']:
        for length in [10, 20]:
            for camera in ['center', 'left', 'right']:
                save_images_to_pcl(data_dir, label, length, camera)
            
            
            
            
            
                
        
        
        
        
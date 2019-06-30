#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from ned_pcl_to_file import import_trajectory_ned


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
        mask = np.array(np.logical_and(np.logical_and(trajectory_ned[0,:]>=xmin,
                                                      trajectory_ned[0,:]<xmax),
                                       np.logical_and(trajectory_ned[1,:]>=ymin, 
                                                      trajectory_ned[1,:]<ymax)))
        mask = mask.squeeze()
        traj_x = np.ravel(trajectory_ned[0, mask])
        traj_y = np.ravel(trajectory_ned[1, mask])
        traj_z = np.ravel(trajectory_ned[2, mask])
        ax.scatter(-traj_y, -traj_x, -traj_z, marker=',', s=1, c='r')
        
    ax.set_xlim(-y_range[1], -y_range[0])
    ax.set_ylim(-x_range[1], -x_range[0])
    ax.set_zlim(-z_range[1], -z_range[0])
    ax.view_init(-60, 0) # elevation, azimuth
    plt.show()


if __name__ == "__main__":
    ins_data_file = 'data/2014-12-02-15-30-08/gps/ins.csv'
    lidar_timestamp_file = 'data/2014-12-02-15-30-08/lms_front.timestamps'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str)
    parser.add_argument('length', type=str)
    parser.add_argument('index', type=str)
    args = parser.parse_args()
    
    pcl_dir = 'pcl/{}_{}m'.format(args.type, args.length)
    submap_filename = pcl_dir + '/submap_2014-12-02_{}_{}m_{}.rawpcl'.format(
                    args.type, args.length, args.index)
    metadata_filename = pcl_dir + '/metadata_2014-12-02_{}_{}m.csv'.format(
                    args.type, args.length)
    
    # Load data 
    submap = np.fromfile(submap_filename, dtype='float64')
    submap = submap.reshape(3, submap.shape[0]//3)
    trajectory_ned = import_trajectory_ned(ins_data_file, lidar_timestamp_file)
    metadata = None
    with open(metadata_filename) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['seg_idx'] == args.index:
                metadata = row
                break
    assert metadata is not None
    
        
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
                
    
    
        
    
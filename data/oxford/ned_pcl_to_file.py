import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import sys


if __name__ == "__main__":
    # Replace with arg parser later if required
    ins_data_file = 'data_sample/gps/ins.csv'
    extrinsics_dir = 'robotcar-dataset-sdk/extrinsics'
    lidar_dir = 'data_sample/lms_front'
    lidar_timestamp_file = 'data_sample/lms_front.timestamps'
    plot = True
    to_file = True
    
    # Get start and end timestamp
    max_frames = 1000
    with open(lidar_timestamp_file) as ts_file:
        start_time = int(next(ts_file).split(' ')[0])
        for it, end_time in enumerate(ts_file):
            if it > max_frames:
                break
            continue
        end_time = int(end_time.split(' ')[0])
    
    # Import the point cloud builder from SDK
    sys.path.insert(0, os.path.join(os.getcwd(),'robotcar-dataset-sdk/python'))
    import build_pointcloud as sdk_pcl
    
    # Calculate point cloud relative to start frame of trajectory
    pointcloud, reflectance = sdk_pcl.build_pointcloud(lidar_dir, ins_data_file,  
                                             extrinsics_dir, start_time, 
                                             end_time)
    
    # Get trajectory corresponding to LIDAR data
    trajectory_ned = np.empty((7,0))
    with open(ins_data_file, 'r') as ins_file:
        reader = csv.DictReader(ins_file)
        for row in reader:
            if float(row['timestamp']) > end_time:
                break
            if float(row['timestamp']) < start_time:
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
    import transform as sdk_trafo
    state_first_frame = ned_state[0:6,0]
    state_first_frame = state_first_frame.flatten()
    state_first_frame[3:6] *= -1
    trafo_veh_ned = sdk_trafo.build_se3_transform(state_first_frame)
    pointcloud_ned = trafo_veh_ned @ pointcloud[0:4,:]
    
    # TODO Trajectory is somehow mirrored to PCL??
    # TODO Split PCL every x meter of trajectory
    
    if to_file:
        pointcloud_ned.tofile('pcl/sample.rawpcl')
 
    if plot:
        x = np.ravel(pointcloud_ned[0, :])
        y = np.ravel(pointcloud_ned[1, :])
        z = np.ravel(pointcloud_ned[2, :])
        traj_x = np.ravel(trajectory_ned[0, :])
        traj_y = np.ravel(trajectory_ned[1, :])
        traj_z = np.ravel(trajectory_ned[2, :])
        colours = (reflectance - reflectance.min()) / (reflectance.max() - reflectance.min())
        colours = 1 / (1 + np.exp(-10 * (colours - colours.mean())))
    
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
        ax.scatter(-y, -x, -z, marker=',', s=1, c=colours, cmap='gray', 
                   edgecolors='none')
        ax.scatter(-traj_y, -traj_x, -traj_z, marker=',', s=1, c='r')
        ax.set_xlim(-y_range[1], -y_range[0])
        ax.set_ylim(-x_range[1], -x_range[0])
        ax.set_zlim(-z_range[1], -z_range[0])
        ax.view_init(140, 0) # elevation, azimuth
        plt.show()

    
    
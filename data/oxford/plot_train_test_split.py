#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from ned_pcl_to_file import import_trajectory_ned

if __name__ == '__main__':
    test_ratio = 0.7
    n_splits = 10
    ins_data_file = 'data/reference/gps/ins.csv'
    lidar_timestamp_file = 'data/reference/lms_front.timestamps'
    trajectory_ned = import_trajectory_ned(ins_data_file, lidar_timestamp_file)
    
    total_length = 0
    prev_pos = trajectory_ned[:3,0]
    for i in range(1, trajectory_ned.shape[1]):
        curr_pos = trajectory_ned[:3,i]
        total_length += np.linalg.norm(prev_pos - curr_pos)
        prev_pos = curr_pos
    
    split_length = total_length / n_splits
    print('Total length {}, split length {}'.format(total_length, split_length))
    
    trajectory_train = np.empty((3,0))
    trajectory_test = np.empty((3,0))
    i_start = 0
    for split_idx in range(n_splits):
        length = 0
        prev_pos = trajectory_ned[:3,i_start]
        for i in range(i_start+1, trajectory_ned.shape[1]):
            curr_pos = trajectory_ned[:3,i]
            length += np.linalg.norm(prev_pos - curr_pos)
            prev_pos = curr_pos
            
            if length > split_length:
                i_start = i
                break
            
            if length < test_ratio*split_length:
                trajectory_train = np.append(trajectory_train, 
                                             curr_pos.reshape(3,1), axis=1)
            else:
                trajectory_test = np.append(trajectory_test, 
                                            curr_pos.reshape(3,1), axis=1)
    
    plt.scatter(trajectory_train[1,:], trajectory_train[0,:], c='green', s=0.5, 
                marker='.')
    plt.scatter(trajectory_test[1,:], trajectory_test[0,:], c='red', s=0.5, 
                marker='.')
    plt.show()
        
        
    
    
    
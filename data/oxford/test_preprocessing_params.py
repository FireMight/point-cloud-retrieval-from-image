#!/usr/bin/env python3
from ned_pcl_to_file import *
from image_preprocessing import *
from pcl_trafo import pcl_trafo

sys.path.insert(0, os.path.join(os.getcwd(),'robotcar-dataset-sdk/python'))
from transform import build_se3_transform


def get_camera_pose():
    with open('robotcar-dataset-sdk/extrinsics/stereo.txt') as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(' ')]

    G_camera_vehicle = build_se3_transform(extrinsics)
    
    with open('robotcar-dataset-sdk/extrinsics/ins.txt') as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_camera_posesource = G_camera_vehicle * build_se3_transform([float(x) for x in extrinsics.split(' ')])

    return G_camera_posesource


if __name__ == "__main__":
    # Replace with arg parser later if required
    ins_data_file = 'data_sample/gps/ins.csv'
    lidar_dir = 'data_sample/lms_front'
    lidar_timestamp_file = 'data_sample/lms_front.timestamps'
    camera_dir = 'data_sample/stereo/centre'
    camera_timestamp_file = 'data_sample/stereo.timestamps'
    
    submap_coverages = [10, 20, 30]
    img_offsets = [0, 1, 2, 5]
    
    pcl_data = import_trajectory_pcl_ned(lidar_timestamp_file, lidar_dir, 
                                         ins_data_file)
    trajectory_ned, pointcloud_ned, reflectance = (pcl_data[0], pcl_data[1], 
                                                   pcl_data[2])
    
    img_data = import_camera_trajectory(camera_timestamp_file, ins_data_file, 
                                        camera_dir)
    camera_trajectory, camera_model = img_data[0], img_data[1]
    
    G_camera_posesource = get_camera_pose()
    
    
    # Pick "random sample" (roughly in the middle of the segment for testing)
    center_idx = 400
    
    
    #### TODO move into loop ####
    coverage = 20
    offset = 2
    
    
    submap, reflectance_seg = get_pcl_segment(trajectory_ned, pointcloud_ned, 
                                              center_idx, coverage, 
                                              alignment='trajectory')
    
    #submap = simple_pcl_downsampling(submap, 4096, 0.1)
    #plot_pcl_traj(submap, reflectance=reflectance_seg)
    
    # Get corresponding image point by going backwards along the trajectory 
    # starting at center idx
    prev_pos = trajectory_ned[[0,1,2], center_idx]
    dist_to_img = coverage/2 + offset
    image_idx = None
    for idx in range(center_idx-1, 0, -1):
        curr_pos = trajectory_ned[[0,1,2],idx]
        dist_to_img = dist_to_img - np.linalg.norm(prev_pos - curr_pos)
        prev_pos = curr_pos
        
        if dist_to_img <= 0:
            image_idx = idx
            break
    
    if image_idx is not None:
        # Get corresponding image
        camera_state = camera_trajectory[:,image_idx]
        image, pil_image = load_image(camera_dir, camera_state[6], camera_model)
        
        # Transform pcl back to vehicle reference system
        ref_state = camera_state[:6]
        ref_state = ref_state.flatten()
        submap_veh = pcl_trafo(submap, trans_oldref=-ref_state[:3],
                               rot=-ref_state[3:])
        
        # Transform pcl into camera frame
        submap_cam = G_camera_posesource @ submap_veh

        # Project points into image
        uv, depth = camera_model.project(submap_cam, image.shape)
        
        # Get ratio of points in image to total points
        points_in_image_ratio = uv.shape[1] / submap.shape[1]

        plt.imshow(pil_image)
        plt.hold(True)
        plt.scatter(np.ravel(uv[0, :]), np.ravel(uv[1, :]), s=2, c=depth, 
                    edgecolors='none', cmap='jet')
        plt.xlim(0, image.shape[1])
        plt.ylim(image.shape[0], 0)
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        print('Image position is beyond start point of trajectory')
        
    
    
    
    
    
    
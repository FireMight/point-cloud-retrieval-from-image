import numpy as np
from scipy.spatial.transform import Rotation

def pcl_trafo(pcl, trans_oldref=np.zeros(3), trans_newref=np.zeros(3), 
              rot=np.zeros(3)):
    R = (Rotation.from_euler('x', rot[0]).as_dcm() @
         Rotation.from_euler('y', rot[1]).as_dcm() @
         Rotation.from_euler('z', rot[2]).as_dcm())
    
    pcl_new = R @ (pcl[:3,:] + trans_oldref.reshape(3,1)) + trans_newref.reshape(3,1)    
    return np.vstack((pcl_new, np.ones((1, pcl_new.shape[1]))))
        
        
    
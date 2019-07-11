#!/usr/bin/env python3
import os
import tarfile


if __name__ == "__main__":
    run_str = None
    
    #required_data = ['lms_front', 'stereo_centre', 'gps', 'tags', 'vo']
    required_data = ['mono_left', 'mono_right']
    tar_archieves = {}
    for name in required_data:
        tar_archieves[name] = []
    
    for file in os.listdir('data'):
        if file.endswith('.tar'):
            # First file: Get name of run
            if run_str is None:
                run_str = file.split('_')[0]
            
            for name in required_data:
                if name in file and run_str in file:
                    tar_archieves[name].append(file)
                    break
    
    # Check if all data is loaded
    for data_type, file_names in tar_archieves.items():
        assert len(file_names) > 0, data_type
        
    # Create root directory for data
    root_dir = 'data/' + run_str
    os.mkdir(root_dir)
    
    # Extract tar files into subdirectories
    for data_type, file_names in tar_archieves.items():
        if data_type == 'stereo_centre':
            data_dir = root_dir + '/stereo'
            os.mkdir(data_dir) # Centre camera has two hierarchy levels...
            data_dir += '/centre'
        else:
            data_dir = root_dir + '/' + data_type
        print('Creating data dir ', data_dir)
        os.mkdir(data_dir)
        
        for chunk_id, file_name in enumerate(file_names):
            print('Extracting {} chunk {}...'.format(data_type, chunk_id+1))
            with tarfile.open('data/' + file_name, 'r') as tar:
                for member in tar.getmembers():
                    if member.isreg():  # skip if the TarInfo is not files
                        member.name = os.path.basename(member.name) # remove the path by reset it
                        tar.extract(member, data_dir) # extract
            
        # Move timestamp files if one exists
        for timestamp_file in [file for file in os.listdir(data_dir) if file.endswith('.timestamps')]:
            print('Moving timestamps file ' + timestamp_file + ' to root dir')
            os.rename(data_dir + '/' + timestamp_file, root_dir + '/' + timestamp_file)
                
    
    
    
    
        
            
    
    
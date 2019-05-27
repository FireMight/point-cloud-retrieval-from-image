import csv
import math
import matplotlib.pyplot as plt



with open('data_sample/gps/ins.csv', 'r') as csv_file:
    reader=csv.DictReader(csv_file)
    
    date = None
    latitude = []
    longitude = []
    dist_travelled = 0.0
    dist_last_split = 0.0
    dist_split = 30.0
    plot_color = 'r'
    n_segments = 0
    
    for i, row in enumerate(reader):
        if i == 0:
            ned_pos = (float(row['northing']),
                       float(row['easting']),
                       float(row['down']))
        if i % 100:
            ned_dist = (ned_pos[0] - float(row['northing']),
                        ned_pos[1] - float(row['easting']),
                        ned_pos[2] - float(row['down']))
            dist_travelled += math.sqrt(ned_dist[0]**2 + 
                                        ned_dist[1]**2 + 
                                        ned_dist[2]**2)
            ned_pos = (float(row['northing']),
                       float(row['easting']),
                       float(row['down']))
            
            # Plot lat, lon every 100m in different color
            latitude.append(float(row['latitude']))
            longitude.append(float(row['longitude']))
            
            if dist_travelled > dist_last_split + dist_split:
                plt.plot(longitude, latitude, color=plot_color)
                n_segments += 1
                dist_last_split = dist_travelled           
                latitude = []
                longitude = []
                
                if plot_color == 'r':
                    plot_color = 'b'
                elif plot_color == 'b':
                    plot_color = 'g'
                else:
                    plot_color = 'r'

print('Number segments:', n_segments)          
plt.show()
    
    
    
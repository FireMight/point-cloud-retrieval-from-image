#include <experimental/random>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/segmentation/sac_segmentation.h>

const std::size_t MAX_NUM_POINTS = 10000000;
const std::size_t DIM = 3;

// threshold used by ground plane removal
const float GROUND_THRESH = 0.75f;

// how many points should be used in the dowsampled submaps
const std::size_t NUM_POINTS_DOWNSAMPLED = 4096;

using vec_t = float;


/**
 * \brief reads in a pcl stored in binary form using numpy's tofile() function and converts
 * it to a pcl::PointCloud<pcl::PointXYZ>
 * 
 * the numeric type (float or double) must be known beforehand - it's specified by vec_t
 *  
 */
pcl::PointCloud<pcl::PointXYZ>::Ptr read_rawpcl(std::string path) {
    std::ifstream infile(path);
    if (infile.good()) {
        infile.seekg(0, infile.end);
        size_t length = infile.tellg();
        infile.seekg(0, infile.beg);
        
        if (length%(sizeof(vec_t)*DIM)!=0) {
            infile.close();
            throw std::logic_error("File corrupt or numerical values of wrong type");
        }
            
        
        std::size_t num_points = length/(sizeof(vec_t)*DIM);

        if (num_points>MAX_NUM_POINTS) {
            infile.close();
            throw std::logic_error("File " + path + " exceeds maximum allowed file size");
        }
            

        std::cout<<"Reading point cloud from file " + path<<std::endl;
        std::cout<<"Cloud consists of "<<num_points<<" points\n";
        std::vector<vec_t> rawpcl(num_points*DIM);
        infile.read((char*)&rawpcl[0], length);
        infile.close();

        //convert to pcl point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcl = boost::make_shared<pcl::PointCloud<pcl::PointXYZ>>(num_points,1);
        for (std::size_t i = 0;i<num_points;i++) {
            (*pcl)[i].x = rawpcl[i];
            (*pcl)[i].y = rawpcl[num_points+i];
            (*pcl)[i].z = rawpcl[2*num_points+i];

            // std::cout<<(*pcl)[i]<<std::endl;
        }
        return pcl;
    }
    else {
        infile.close();
        throw std::logic_error("Couldn't open file " + path);
    }
}

/**
 * \brief writes a pcl::PointCloud to the file specified by path
 * 
 * numeric type is specified by vec_t
 */
void write_rawpcl(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::string path) {
    std::ofstream outfile(path);
    if(outfile.good()) {
        // serialize cloud in a vec_t vector
        std::size_t num_points = cloud->width;
        std::vector<vec_t> rawpcl(num_points*DIM);
        for (std::size_t i=0;i<num_points;i++) {
            rawpcl[i] = (*cloud)[i].x;
            rawpcl[num_points+i] = (*cloud)[i].y;
            rawpcl[2*num_points+i] = (*cloud)[i].z;
        }

        outfile.write((char*)&rawpcl[0], num_points*DIM*sizeof(vec_t));
        outfile.close();
        std::cout<<"Point cloud written to "<<path<<std::endl;
        outfile.close();
    }
    else {
        outfile.close();
        throw std::logic_error("Couldn't open file for writing " + path);
    }
}

int main(int argc, char* argv[]) {
    using namespace pcl;

    std::string path = "../pcl/sample.rawpcl";
    bool show_vis = false;
    std::string output_path = "";

    if(argc>1) {
        path = argv[1];
    }
    if(argc>2) {
        output_path = argv[2];
    } 
    if(argc>3) {
        show_vis = std::atoi(argv[3]);
    } 
    auto raw_cloud = read_rawpcl(path);

    ModelCoefficients plane;
    plane.values.resize(4);
    auto plane_indices = boost::make_shared<PointIndices>();

    // remove ground plane
    SACSegmentation<PointXYZ> seg;
    seg.setOptimizeCoefficients(true);
    seg.setMethodType(SAC_RANSAC);
    seg.setModelType(SACMODEL_PLANE);
    seg.setDistanceThreshold(GROUND_THRESH);
    seg.setInputCloud(raw_cloud);
    seg.segment(*plane_indices,plane);

    if (plane_indices->indices.size() == 0) {
		PCL_ERROR ("Could not estimate a planar model for the given point cloud.\n");
		return (-1);
	}
    else {
        std::cout<<plane_indices->indices.size()<<" points classified as ground and removed\n";
    }

    auto no_ground = boost::make_shared<PointCloud<PointXYZ>>();
    auto ground = boost::make_shared<PointCloud<PointXYZ>>();
    ExtractIndices<PointXYZ> extract;
    extract.setInputCloud(raw_cloud);
    extract.setIndices(plane_indices);
    extract.setNegative(true);
    extract.filter(*no_ground);

    extract.setNegative(false);
    extract.filter(*ground);


    // downsample using a voxel grid filter
    if(no_ground->width<NUM_POINTS_DOWNSAMPLED) {
        throw std::logic_error("Point cloud not dense enough");
    }

    // estimate starting leaf size for the number of downsampled points
    auto eigen_map = no_ground->getMatrixXfMap(3,4,0);
    float volume = 1; 
    for (int i=0;i<3;i++) {
        float max = eigen_map.row(i).maxCoeff();
        float min = eigen_map.row(i).minCoeff();
        volume *= max-min;
    }
    float leaf_size = std::pow(volume/(NUM_POINTS_DOWNSAMPLED*27), 1.0/3);
    
    auto downsampled = boost::make_shared<PointCloud<PointXYZ>>();
    VoxelGrid<PointXYZ> filter;
    filter.setInputCloud(no_ground);

    std::size_t max_attemps = 20;
    // try downsampling with different leaf sizes until we get approximately NUM_POINTS_DOWNSAMPLED points
    for (std::size_t i = 0; i<max_attemps;i++)
    {   
        std::cout<<"Downsampling with a leaf size of "<<leaf_size<<std::endl;
        filter.setLeafSize(leaf_size,leaf_size,leaf_size);
        filter.filter(*downsampled);

        std::cout<<"Downsampled cloud contains "<<downsampled->width<<" points"<<std::endl;
        float factor = downsampled->width*1.0/NUM_POINTS_DOWNSAMPLED;
        if (factor<0.97 || factor>1.03) {
            // estimate new leaf size
            leaf_size = std::pow(factor,1.0/3)*leaf_size;
            std::cout<<"Modifying leaf size and trying again"<<std::endl;
        }
        else break;
        
        if (i==max_attemps-1) {
            throw std::logic_error("Couldn't downsample cloud to the required number of points");
        }
    }
    // add or remove points from dowsampled cloud to get to exactly NUM_POINTS_DOWNSAMPLED
    while(downsampled->width>NUM_POINTS_DOWNSAMPLED) {
        int i = std::experimental::randint(0,(int)downsampled->width-1);
        auto iter = downsampled->begin() + i;
        downsampled->erase(iter);
    }
    while(downsampled->width<NUM_POINTS_DOWNSAMPLED) {
        int i = std::experimental::randint(0,(int)no_ground->width-1);
        downsampled->push_back( (*no_ground)[i] );
    }


    // write processed pcl
    std::size_t ext_index = path.find_last_of("/");
    if (ext_index==-1) ext_index = 0;
    else ext_index++;
    path = output_path + path.substr(ext_index) + ".processed";
    write_rawpcl(downsampled,path);

    if(show_vis) {
        std::cout<<"Starting visualiazer\n";
        visualization::PCLVisualizer viewer("PCL visualizer");
        visualization::PointCloudColorHandlerCustom<PointXYZ> ground_handler(ground,255,99,71);
        viewer.addPointCloud(no_ground);
        viewer.addPointCloud(ground,ground_handler,"ground");
        // viewer.setCameraPosition(5735235.0,620191,-236.468,0,0,0);
        // viewer.resetCameraViewpoint("ground");

        while(!viewer.wasStopped()) {
            viewer.spinOnce();
        }
        
        viewer.removeAllPointClouds();
        viewer.addPointCloud(downsampled);
        
        viewer.resetStoppedFlag();
        while(!viewer.wasStopped()) {
            viewer.spinOnce();
        }
        
    }

    return 0;
}
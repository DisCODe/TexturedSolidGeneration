/*!
 * \file
 * \brief
 * \author Michal Laszkowski
 */

#include <memory>
#include <string>

#include "CuboidModelGenerator.hpp"
#include "Common/Logger.hpp"

#include <boost/bind.hpp>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#if (CV_MAJOR_VERSION == 2)
#if (CV_MINOR_VERSION > 3)
#include <opencv2/nonfree/features2d.hpp>
#endif
#endif

// Maximal size of a single side cloud.
#define max_cloud_size 1000000

using boost::property_tree::ptree;
using boost::property_tree::read_json;

namespace Processors {
namespace CuboidModelGenerator {

CuboidModelGenerator::CuboidModelGenerator(const std::string & name) :
		Base::Component(name) , 
        dataJSONname("dataJSONname", std::string("./")),
        generate_on_init("generate_on_init", true),
        resolution("resolution", 1.0){
    registerProperty(dataJSONname);
    registerProperty(generate_on_init);
    registerProperty(resolution);
    generateModel_flag = generate_on_init;
}

CuboidModelGenerator::~CuboidModelGenerator() {
}

void CuboidModelGenerator::prepareInterface() {
	// Register data streams, events and event handlers HERE!
	registerStream("out_cloud_xyzrgb", &out_cloud_xyzrgb);
    registerStream("out_cloud_xyzsift", &out_cloud_xyzsift);

    // Register handlers
    h_generateModel.setup(boost::bind(&CuboidModelGenerator::generateModelPressed, this));
    registerHandler("generateModel", &h_generateModel);

    h_returnModel.setup(boost::bind(&CuboidModelGenerator::returnModel, this));
    registerHandler("returnModel", &h_returnModel);
    addDependency("returnModel", NULL);
}

bool CuboidModelGenerator::onInit() {
    CLOG(LTRACE) << "CuboidModelGenerator::onInit";
    generate_top = generate_bottom = generate_left = generate_right = generate_front = generate_back =
        mask_top = mask_bottom = mask_left = mask_right = mask_front = mask_back = false;
    cloud_xyzrgb = pcl::PointCloud<pcl::PointXYZRGB>::Ptr (new pcl::PointCloud<pcl::PointXYZRGB>());
    cloud_xyzsift = pcl::PointCloud<PointXYZSIFT>::Ptr (new pcl::PointCloud<PointXYZSIFT>());
	return true;
}

bool CuboidModelGenerator::onFinish() {
	return true;
}

bool CuboidModelGenerator::onStop() {
	return true;
}

bool CuboidModelGenerator::onStart() {
	return true;
}

void CuboidModelGenerator::sift(cv::Mat input, cv::Mat &descriptors, Types::Features &features) {
    CLOG(LTRACE) << "CuboidModelGenerator::sift";
    try {
        //-- resolution 1: Detect the keypoints.
        cv::SiftFeatureDetector detector;
        std::vector<cv::KeyPoint> keypoints;
        detector.detect(input, keypoints);

        //-- resolution 2: Calculate descriptors (feature vectors).
        cv::SiftDescriptorExtractor extractor;
        extractor.compute( input, keypoints, descriptors);

        features = Types::Features(keypoints);
    } catch (...) {
        CLOG(LERROR) << "CuboidModelGenerator::sift() failed\n";
    }
}

std::string dirnameOf(const std::string& fname)
{
     size_t pos = fname.find_last_of("\\/");
     return (std::string::npos == pos)
         ? ""
         : fname.substr(0, pos);
}

void CuboidModelGenerator::loadData(){
    CLOG(LTRACE) << "CuboidModelGenerator::loadData";
    ptree ptree_file;
    std::string model_name;
    std::string left_name;
    std::string right_name;
    std::string top_name;
    std::string bottom_name;
    std::string front_name;
    std::string back_name;
    std::string left_mask_name;
    std::string right_mask_name;
    std::string top_mask_name;
    std::string bottom_mask_name;
    std::string front_mask_name;
    std::string back_mask_name;
    std::string dir;
    try{
        // Open JSON file and load it to ptree.
        read_json(dataJSONname, ptree_file);
    }//: catch
    catch(std::exception const& e){
        CLOG(LERROR) << "CuboidModelGenerator: file "<< dataJSONname <<" not found";
        return;
    }//: catch
    try{
    	// Get path to files.
        dir = dirnameOf(dataJSONname);
        if (dir!="")
        	dir = dir + "/";
        //cout<< dir <<endl;
        // Read JSON properties.
        model_name = ptree_file.get("name","");
        left_name = ptree_file.get("left","");
        right_name = ptree_file.get("right","");
        top_name = ptree_file.get("top","");
        bottom_name = ptree_file.get("bottom","");
        front_name = ptree_file.get("front","");
        back_name = ptree_file.get("back","");
        left_mask_name = ptree_file.get("left_mask","");
        right_mask_name = ptree_file.get("right_mask","");
        top_mask_name = ptree_file.get("top_mask","");
        bottom_mask_name = ptree_file.get("bottom_mask","");
        front_mask_name = ptree_file.get("front_mask","");
        back_mask_name = ptree_file.get("back_mask","");

        width = ptree_file.get<int>("width");
        depth = ptree_file.get<int>("depth");
        height = ptree_file.get<int>("height");
    }//: try
    catch(std::exception const& e){
        CLOG(LERROR) << "CuboidModelGenerator: file "<< dataJSONname <<" invalid";
        return;
    }//: catch
    try{
    if(left_name!=""){
    	//cout<< (std::string)(dir + left_name) <<endl;
        left = cv::imread((std::string)(dir + left_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        generate_left = true;
    }
    if(right_name!=""){
        right = cv::imread((std::string)(dir + right_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        generate_right = true;
    }
    if(top_name!=""){
        top = cv::imread((std::string)(dir + top_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        generate_top = true;
    }
    if(bottom_name!=""){
        bottom = cv::imread((std::string)(dir + bottom_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        generate_bottom = true;
    }
    if(front_name!=""){
        front = cv::imread((std::string)(dir + front_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        generate_front =  true;
    }
    if(back_name!=""){
        back = cv::imread((std::string)(dir + back_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        generate_back = true;
    }
    if(left_mask_name!=""){
        left_mask = cv::imread((std::string)(dir + left_mask_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        mask_left = true;
    }
    if(right_mask_name!=""){
        right_mask = cv::imread((std::string)(dir + right_mask_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        mask_right = true;
    }
    if(top_mask_name!=""){
        top_mask = cv::imread((std::string)(dir + top_mask_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        mask_top = true;
    }
    if(bottom_mask_name!=""){
        bottom_mask = cv::imread((std::string)(dir + bottom_mask_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        mask_bottom = true;
    }
    if(front_mask_name!=""){
        front_mask = cv::imread((std::string)(dir + front_mask_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        mask_front =  true;
    }
    if(back_mask_name!=""){
        back_mask = cv::imread((std::string)(dir + back_mask_name), CV_LOAD_IMAGE_ANYDEPTH | CV_LOAD_IMAGE_ANYCOLOR);
        mask_back = true;
    }
    }//: try
    catch(std::exception const& e){
    	CLOG(LERROR) << "CuboidModelGenerator: invalid files with textures";
        return;
    }//: catch
}


void CuboidModelGenerator::returnModel() {
    CLOG(LTRACE) << "CuboidModelGenerator::returnModel()";
    // Generate model if required.
    if (generateModel_flag) {
        generateModel();
        generateModel_flag = false;
    }

    // Write to output clouds and SOM.
    out_cloud_xyzrgb.write(cloud_xyzrgb);
    out_cloud_xyzsift.write(cloud_xyzsift);
}


void CuboidModelGenerator::generateModelPressed() {
    CLOG(LTRACE) << "CuboidModelGenerator::generateModelPressed";
    generateModel_flag = true;
}


void CuboidModelGenerator::generateModel() {
    CLOG(LTRACE) << "CuboidModelGenerator::generateModel";
    loadData();

    // Clear clouds.
    cloud_xyzrgb->clear();
    cloud_xyzsift->clear();

    // Check desired size of point cloud.
    int front_size = width*resolution * height*resolution;
    int bottom_size = width*resolution * depth*resolution;
    int side_size = depth*resolution * height*resolution;
    CLOG(LNOTICE) <<"Size of clouds: front=" << front_size << " bottom=" << bottom_size << "side=" << side_size;
    if ((front_size > max_cloud_size) || (bottom_size > max_cloud_size) || (side_size > max_cloud_size)) {
    	CLOG(LERROR) << "Maximal size of a cloud of one cuboid side (10^6 points) exceeded";
    	return;
    }

#if 1
    //******************************************************************************************
    // FRONT side of cuboid.
    if(generate_front){
        // Create image with reduced (rescaled) dimensions.
        cv::Mat rescaled_texture(cv::Size((int)(width*resolution), (int)(height*resolution)), CV_8UC3);

        CLOG(LTRACE) <<"FRONT " << front.cols << " x " <<front.rows << endl;

        // Generate side on XZ plane.
        for(float x = 0; x < width*resolution; x+=1){
            for(float z = 0; z < height*resolution; z+=1){
                // Compute image coordinates.
                int xx = (x/resolution)*(front.cols)/(width);
                int zz = (z/resolution)*(front.rows)/(height);
                // Get colour of the original image point.
                cv::Vec3b bgr = front.at<cv::Vec3b>(zz, xx);

                // Set colour of the rescaled image point.
                CLOG(LDEBUG)<<rescaled_texture.rows <<" x "<<rescaled_texture.cols<<" coord: "<<x<<","<<z<<endl;
                rescaled_texture.at<cv::Vec3b>(z, x) = cv::Vec3b(bgr[0], bgr[1], bgr[2]);

                // Create point with cartesian coordinates.
                // Skip points not lying under mask - if mask was loaded.
                if (mask_front && front_mask.at<float>(zz,xx)==0)
                        continue;
                pcl::PointXYZRGB point;
                // Set point cartesian coordinates.
                point.x = (float(width)-x/resolution)/1000;
                point.y = 0;
                point.z = (float(height)-z/resolution)/1000;
                // Set point colours.
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                // Add point to cloud.
                cloud_xyzrgb->push_back(point);

            }// for z
        }// for x

        cv::Mat descriptors;
		Types::Features features;
		// Detect features in the rescaled image.
		sift(rescaled_texture, descriptors, features);
		CLOG(LTRACE) << "SIFT FRONT " << features.features.size() << endl;
		// Iterate on detected features.
		for (int i = 0; i < features.features.size(); i++) {
            // Skip points not lying under mask - if mask was loaded - TODO!
/*           int u = round(features.features[i].pt.x * (front.rows/height));
			 int v = round(features.features[i].pt.y * (front.cols/width));
			 if (mask_front && front_mask.at<float>(v, u)==0) {
			 continue;
			 }
*/

            // Create SIFT point with cartesian coordinates.
			PointXYZSIFT point;
			point.x = (float(width) - float(features.features[i].pt.x) / resolution) / 1000;
			point.y = 0;
			point.z = (float(height) - float(features.features[i].pt.y) / resolution) / 1000;
			// Copy descriptor.
			for (int j = 0; j < descriptors.cols; j++) {
				point.descriptor[j] = descriptors.row(i).at<float>(j);
			}
			point.multiplicity = 1;
			// Add point to cloud.
			cloud_xyzsift->push_back(point);
        }// for
    }// if FRONT
#endif

#if 1
    //******************************************************************************************
    // BACK side of cuboid.
    if(generate_back){
        // Create image with reduced (rescaled) dimensions.
        cv::Mat rescaled_texture(cv::Size((int)(width*resolution), (int)(height*resolution)), CV_8UC3);

        CLOG(LTRACE) <<"BACK " << back.cols << " x " <<back.rows << endl;

        // Generate side on XZ plane.
        for(float x = 0; x < width*resolution; x+=1){
            for(float z = 0; z < height*resolution; z+=1){
                // Compute image coordinates.
                int xx = (x/resolution)*(back.cols)/(width);
                int zz = (z/resolution)*(back.rows)/(height);

                // Get colour of the original image point.
                cv::Vec3b bgr = back.at<cv::Vec3b>(zz, xx);

                // Set colour of the rescaled image point.
                CLOG(LDEBUG)<<rescaled_texture.rows <<" x "<<rescaled_texture.cols<<" coord: "<<x<<","<<z<<endl;
                rescaled_texture.at<cv::Vec3b>(z, x) = cv::Vec3b(bgr[0], bgr[1], bgr[2]);

                // Create point with cartesian coordinates.
                // Skip points not lying under mask - if mask was loaded.
                if (mask_back && back_mask.at<float>(zz,xx)==0)
                        continue;
                pcl::PointXYZRGB point;
                // Set point cartesian coordinates.
                point.x = (x/resolution)/1000;
                point.y = -float(depth)/1000;
                point.z = (float(height)-z/resolution)/1000;
                // Set point colours.
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                // Add point to cloud.
                cloud_xyzrgb->push_back(point);

            }// for z
        }// for x

        cv::Mat descriptors;
		Types::Features features;
		// Detect features in the rescaled image.
		sift(rescaled_texture, descriptors, features);
		CLOG(LTRACE) << "SIFT BACK " << features.features.size() << endl;
		// Iterate on detected features.
		for (int i = 0; i < features.features.size(); i++) {
            // Skip points not lying under mask - if mask was loaded - TODO!
			// ...

            // Create SIFT point with cartesian coordinates.
			PointXYZSIFT point;
            point.x = (float(features.features[i].pt.x)/resolution)/1000;
            point.y = -float(depth)/1000;
            point.z = (float(height) - float(features.features[i].pt.y)/resolution)/1000;

			// Copy descriptor.
			for (int j = 0; j < descriptors.cols; j++) {
				point.descriptor[j] = descriptors.row(i).at<float>(j);
			}
			point.multiplicity = 1;
			// Add point to cloud.
			cloud_xyzsift->push_back(point);
        }// for
    }// if BACK

#endif

#if 1
    //******************************************************************************************
    // TOP side of cuboid.
    if(generate_top){
        // Create image with reduced (rescaled) dimensions.
        cv::Mat rescaled_texture(cv::Size((int)(width*resolution), (int)(depth*resolution)), CV_8UC3);

        CLOG(LTRACE) <<"TOP " << top.cols << " x " <<top.rows << endl;

        // Generate side on XZ plane.
        for(float x = 0; x < width*resolution; x+=1){
            for(float y = 0; y < depth*resolution; y+=1){
                // Compute image coordinates.
                int xx = (x/resolution)*(top.cols)/(width);
                int yy = (y/resolution)*(top.rows)/(depth);

                // Get colour of the original image point.
                cv::Vec3b bgr = top.at<cv::Vec3b>(yy, xx);

                // Set colour of the rescaled image point.
                CLOG(LDEBUG)<<rescaled_texture.rows <<" x "<<rescaled_texture.cols<<" coord: "<<x<<","<<y<<endl;
                rescaled_texture.at<cv::Vec3b>(y, x) = cv::Vec3b(bgr[0], bgr[1], bgr[2]);

                // Create point with cartesian coordinates.
                // Skip points not lying under mask - if mask was loaded.
                if (mask_top && top_mask.at<float>(yy,xx)==0)
                        continue;
                pcl::PointXYZRGB point;
                // Set point cartesian coordinates.
                point.x = (float(width) - x/resolution)/1000;
                point.y = (float(-depth) + y/resolution)/1000;
                point.z = float(height)/1000;
                // Set point colours.
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                // Add point to cloud.
                cloud_xyzrgb->push_back(point);

            }// for y
        }// for x

        cv::Mat descriptors;
		Types::Features features;
		// Detect features in the rescaled image.
		sift(rescaled_texture, descriptors, features);
		CLOG(LTRACE) << "SIFT TOP " << features.features.size() << endl;
		// Iterate on detected features.
		for (int i = 0; i < features.features.size(); i++) {
            // Skip points not lying under mask - if mask was loaded - TODO!
			// ...

			// Create SIFT point with cartesian coordinates.
			PointXYZSIFT point;
            point.x = (float(width) - float(features.features[i].pt.x)/resolution)/1000;
            point.y = (float(-depth) + float(features.features[i].pt.y)/resolution)/1000;
            point.z = float(height)/1000;

			// Copy descriptor.
			for (int j = 0; j < descriptors.cols; j++) {
				point.descriptor[j] = descriptors.row(i).at<float>(j);
			}
			point.multiplicity = 1;
			// Add point to cloud.
			cloud_xyzsift->push_back(point);
        }// for
    }// if TOP
#endif

#if 1
    //******************************************************************************************
    // BOTTOM side of cuboid.
    if(generate_bottom){
        // Create image with reduced (rescaled) dimensions.
        cv::Mat rescaled_texture(cv::Size((int)(width*resolution), (int)(depth*resolution)), CV_8UC3);

        CLOG(LTRACE) <<"BOTTOM " << bottom.cols << " x " <<bottom.rows << endl;

        // Generate side on XZ plane.
        for(float x = 0; x < width*resolution; x+=1){
            for(float y = 0; y < depth*resolution; y+=1){
                // Compute image coordinates.
                int xx = (x/resolution)*(bottom.cols)/(width);
                int yy = (y/resolution)*(bottom.rows)/(depth);

                // Get colour of the original image point.
                cv::Vec3b bgr = bottom.at<cv::Vec3b>(yy, xx);

                // Set colour of the rescaled image point.
                CLOG(LDEBUG)<<rescaled_texture.rows <<" x "<<rescaled_texture.cols<<" coord: "<<x<<","<<y<<endl;
                rescaled_texture.at<cv::Vec3b>(y, x) = cv::Vec3b(bgr[0], bgr[1], bgr[2]);

                // Create point with cartesian coordinates.
                // Skip points not lying under mask - if mask was loaded.
                if (mask_bottom && bottom_mask.at<float>(yy,xx)==0)
                        continue;

                pcl::PointXYZRGB point;
                // Set point cartesian coordinates.
                point.x = (float(width) - x/resolution)/1000;
                point.y = (float(-depth) + y/resolution)/1000;
                point.z = 0;
                // Set point colours.
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                // Add point to cloud.
                cloud_xyzrgb->push_back(point);

            }// for y
        }// for x

        cv::Mat descriptors;
		Types::Features features;
		// Detect features in the rescaled image.
		sift(rescaled_texture, descriptors, features);
		CLOG(LTRACE) << "SIFT BOTTOM " << features.features.size() << endl;
		// Iterate on detected features.
		for (int i = 0; i < features.features.size(); i++) {
            // Skip points not lying under mask - if mask was loaded - TODO!
			// ...

            // Create SIFT point with cartesian coordinates.
			PointXYZSIFT point;
            point.x = (float(width) - float(features.features[i].pt.x)/resolution)/1000;
            point.y = (float(-depth) + float(features.features[i].pt.y)/resolution)/1000;
            point.z = 0;

			// Copy descriptor.
			for (int j = 0; j < descriptors.cols; j++) {
				point.descriptor[j] = descriptors.row(i).at<float>(j);
			}
			point.multiplicity = 1;
			// Add point to cloud.
			cloud_xyzsift->push_back(point);
        }// for
    }// if BOTTOM
#endif

#if 1
    //******************************************************************************************
    // LEFT side of cuboid.
    if(generate_left){
        // Create image with reduced (rescaled) dimensions.
        cv::Mat rescaled_texture(cv::Size((int)(depth*resolution), (int)(height*resolution)), CV_8UC3);

        CLOG(LTRACE) <<"LEFT " << left.cols << " x " <<left.rows << endl;

        // Generate side on XZ plane.
        for(float y = 0; y < depth*resolution; y+=1){
            for(float z = 0; z < height*resolution; z+=1){
                // Compute image coordinates.
                int yy = (y/resolution)*(left.cols)/(depth);
                int zz = (z/resolution)*(left.rows)/(height);

                // Get colour of the original image point.
                cv::Vec3b bgr = left.at<cv::Vec3b>(zz, yy);

                // Set colour of the rescaled image point.
                CLOG(LDEBUG)<<rescaled_texture.cols <<" x "<<rescaled_texture.rows<<" coord: "<<z<<","<<y<<endl;
                rescaled_texture.at<cv::Vec3b>(z, y) = cv::Vec3b(bgr[0], bgr[1], bgr[2]);

                // Create point with cartesian coordinates.
                // Skip points not lying under mask - if mask was loaded.
                if (mask_left && left_mask.at<float>(zz,yy)==0)
                        continue;

                pcl::PointXYZRGB point;
                // Set point cartesian coordinates.
                point.x = float(width)/1000;
                point.y = (float(-depth) + y/resolution)/1000;
                point.z = (float(height) - z/resolution)/1000;
                // Set point colours.
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                // Add point to cloud.
                cloud_xyzrgb->push_back(point);

            }// for y
        }// for x

        cv::Mat descriptors;
		Types::Features features;
		// Detect features in the rescaled image.
		sift(rescaled_texture, descriptors, features);
		CLOG(LTRACE) << "SIFT LEFT " << features.features.size() << endl;
		// Iterate on detected features.
		for (int i = 0; i < features.features.size(); i++) {
            // Skip points not lying under mask - if mask was loaded - TODO!
			// ...

            // Create SIFT point with cartesian coordinates.
			PointXYZSIFT point;
            point.x = float(width)/1000;
            point.y = (float(-depth) + float(features.features[i].pt.x)/resolution)/1000;
            point.z = (float(height) - float(features.features[i].pt.y)/resolution)/1000;

			// Copy descriptor.
			for (int j = 0; j < descriptors.cols; j++) {
				point.descriptor[j] = descriptors.row(i).at<float>(j);
			}
			point.multiplicity = 1;
			// Add point to cloud.
			cloud_xyzsift->push_back(point);
        }// for
    }// if LEFT
#endif


#if 1
    //******************************************************************************************
    // RIGHT side of cuboid.
    if(generate_right){
        // Create image with reduced (rescaled) dimensions.
        cv::Mat rescaled_texture(cv::Size((int)(depth*resolution), (int)(height*resolution)), CV_8UC3);

        CLOG(LTRACE) <<"RIGHT " << right.cols << " x " <<right.rows << endl;

        // Generate side on XZ plane.
        for(float y = 0; y < depth*resolution; y+=1){
            for(float z = 0; z < height*resolution; z+=1){
                // Compute image coordinates.
                int yy = (y/resolution)*(right.cols)/(depth);
                int zz = (z/resolution)*(right.rows)/(height);

                // Get colour of the original image point.
                cv::Vec3b bgr = right.at<cv::Vec3b>(zz, yy);

                // Set colour of the rescaled image point.
                CLOG(LDEBUG)<<rescaled_texture.cols <<" x "<<rescaled_texture.rows<<" coord: "<<z<<","<<y<<endl;
                rescaled_texture.at<cv::Vec3b>(z, y) = cv::Vec3b(bgr[0], bgr[1], bgr[2]);

                // Create point with cartesian coordinates.
                // Skip points not lying under mask - if mask was loaded.
                if (mask_right && right_mask.at<float>(zz,yy)==0)
                        continue;

                pcl::PointXYZRGB point;
                // Set point cartesian coordinates.
                point.x = 0;
                point.y = (- y/resolution)/1000;
                point.z = (float(height) - z/resolution)/1000;
                // Set point colours.
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                // Add point to cloud.
                cloud_xyzrgb->push_back(point);

            }// for y
        }// for x

        cv::Mat descriptors;
		Types::Features features;
		// Detect features in the rescaled image.
		sift(rescaled_texture, descriptors, features);
		CLOG(LTRACE) << "SIFT RIGHT " << features.features.size() << endl;
		// Iterate on detected features.
		for (int i = 0; i < features.features.size(); i++) {
            // Skip points not lying under mask - if mask was loaded - TODO!
			// ...

            // Create SIFT point with cartesian coordinates.
			PointXYZSIFT point;
            point.x = 0;
            point.y = (- float(features.features[i].pt.x)/resolution)/1000;
            point.z = (float(height) - float(features.features[i].pt.y)/resolution)/1000;

			// Copy descriptor.
			for (int j = 0; j < descriptors.cols; j++) {
				point.descriptor[j] = descriptors.row(i).at<float>(j);
			}
			point.multiplicity = 1;
			// Add point to cloud.
			cloud_xyzsift->push_back(point);
        }// for
    }// if RIGHT
#endif

    CLOG(LNOTICE) <<"Total size of point cloud of cuboid: " << cloud_xyzrgb->size();
    CLOG(LNOTICE) <<"Total number of features: " << cloud_xyzsift->size();

}



} //: namespace CuboidModelGenerator
} //: namespace Processors

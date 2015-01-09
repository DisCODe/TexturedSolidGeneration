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
        cout<< dir <<endl;
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
    	cout<< (std::string)(dir + left_name) <<endl;
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

    float x,y,z;
    //front
    if(generate_front){
        y=0;//stałe
        CLOG(LTRACE) <<"front " << front.cols << " x " <<front.rows << endl;
        for(x = 0; x < width; x+=resolution){
            for(z = 0; z < height; z+=resolution){
                pcl::PointXYZRGB point;
                point.x = (float(width)-x)/1000;
                point.y = y/1000;
                point.z = (float(height)-z)/1000;
                //pozycja w obrazie
                int xx = (x)*(front.cols)/(width);
                int zz = (z)*(front.rows)/(height);
                if (mask_front && front_mask.at<float>(zz,xx)==0) {
                        continue;
                }
                cv::Vec3b bgr = front.at<cv::Vec3b>(zz, xx);
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                cloud_xyzrgb->push_back(point);
            }

        }
    }
    //back
    if(generate_back){
        y=-depth;//stale
        CLOG(LTRACE) <<"back " << back.cols << " x " <<back.rows << endl;
        for(x = 0; x < width; x+=resolution){
            for(z = 0; z < height; z+=resolution){
                pcl::PointXYZRGB point;
                point.x = x/1000;
                point.y = y/1000;
                point.z = (float(height)-z)/1000;
                int xx = x*(back.cols)/(width);
                int zz = z*(back.rows)/(height);
                if (mask_back && back_mask.at<float>(zz, xx)==0) {
                        continue;
                }
                cv::Vec3b bgr = back.at<cv::Vec3b>(zz, xx);
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                cloud_xyzrgb->push_back(point);
            }
        }
    }
    //top
    if(generate_top){
        z= height;//stale
        CLOG(LTRACE) <<"top " << top.cols << " x " <<top.rows << endl;
        for(x = 0; x < width; x+=resolution){
            for(y = 0; y < depth; y+=resolution){
                pcl::PointXYZRGB point;
                point.x = (float(width)-x)/1000;
                point.y = (float(-depth)+y)/1000;
                point.z = z/1000;
                int xx = (x)*(top.cols)/(width);
                int yy = (y)*(top.rows)/(depth);
                if (mask_top && top_mask.at<float>(yy, xx)==0) {
                        continue;
                }
                cv::Vec3b bgr = top.at<cv::Vec3b>(yy, xx);
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                cloud_xyzrgb->push_back(point);
            }
        }
    }
    //bottom
    if(generate_bottom){
        z= 0;//stale
        CLOG(LTRACE) <<"bottom " << bottom.cols << " x " <<bottom.rows << endl;
        for(x = 0; x < width; x+=resolution){
            for(y = 0; y < depth; y+=resolution){
                pcl::PointXYZRGB point;
                //point.x = x/1000;
                point.x = (float(width)-x)/1000;
                point.y = (float(-depth)+y)/1000;
                point.z = z/1000;
                int xx = (x)*(bottom.cols)/(width);
                int yy = (y)*(bottom.rows)/(depth);
                if (mask_bottom && bottom_mask.at<float>(yy, xx)==0) {
                        continue;
                }
                cv::Vec3b bgr = bottom.at<cv::Vec3b>(yy, xx);
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                cloud_xyzrgb->push_back(point);
            }
        }
    }
    //left
    if(generate_left){
        x= width;//stale
        CLOG(LTRACE) <<"left " << left.cols << " x " <<left.rows << endl;
        for(y = 0; y < depth; y+=resolution){
            for(z = 0; z < height; z+=resolution){
                pcl::PointXYZRGB point;
                point.x = x/1000;
                point.y = (float(-depth)+y)/1000;
                point.z = (float(height)-z)/1000;
                int yy = (y)*(left.cols)/(depth);
                int zz = (z)*(left.rows)/(height);
                if (mask_left && left_mask.at<float>(zz, yy)==0) {
                        continue;
                }
                cv::Vec3b bgr = left.at<cv::Vec3b>(zz, yy);
                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                cloud_xyzrgb->push_back(point);
            }
        }
    }
    //right
    if(generate_right){
        x= 0;//stale
        CLOG(LTRACE) <<"right " << right.cols << " x " <<right.rows << endl;
        for(y = 0; y < depth; y+=resolution){
            for(z = 0; z < height; z+=resolution){
                pcl::PointXYZRGB point;
                point.x = x/1000;
                point.y = -y/1000;
                point.z = (float(height)-z)/1000;
                int yy = y*(right.cols)/(depth);
                int zz = z*(right.rows)/(height);
                if (mask_right && right_mask.at<float>(zz, yy)==0) {
                        continue;
                }
                cv::Vec3b bgr = right.at<cv::Vec3b>(zz, yy);

                point.r = bgr[2];
                point.g = bgr[1];
                point.b = bgr[0];
                cloud_xyzrgb->push_back(point);
            }
        }
    }

#if 0
    //SIFT
    cv::Mat descriptors;
    Types::Features features;
    //front
    if(generate_front){
        sift(front,descriptors,features);
        CLOG(LTRACE)<<"SIFT front " << features.features.size() <<endl;
        for(int i=0; i < features.features.size(); i++){
            PointXYZSIFT point;
            int u = round(features.features[i].pt.x);
            int v = round(features.features[i].pt.y);
            if (mask_front && front_mask.at<float>(v, u)==0) {
                    continue;
            }

            int xx = (u)*(width)/(front.cols);
            int zz = (v)*(height)/(front.rows);

            point.x = float(width-xx)/1000;
            point.y = float(0)/1000;
            point.z = float(height-zz)/1000;
            for(int j=0; j<descriptors.cols;j++){
                point.descriptor[j] = descriptors.row(i).at<float>(j);
            }
            point.multiplicity = 1;
            cloud_xyzsift->push_back(point);
        }
    }
    //back
    if(generate_back){
        sift(back,descriptors,features);
        CLOG(LTRACE)<<"SIFT back " << features.features.size() <<endl;
        for(int i=0; i < features.features.size(); i++){
            PointXYZSIFT point;
            int u = round(features.features[i].pt.x);
            int v = round(features.features[i].pt.y);
            if (mask_back && back_mask.at<float>(v, u)==0) {
                    continue;
            }

            int xx = (u)*(width)/(back.cols);
            int zz = (v)*(height)/(back.rows);

            point.x = float(xx)/1000;
            point.y = float(-depth)/1000;
            point.z = float(height-zz)/1000;
            for(int j=0; j<descriptors.cols;j++){
                point.descriptor[j] = descriptors.row(i).at<float>(j);
            }
            point.multiplicity = 1;
            cloud_xyzsift->push_back(point);
        }
    }

    //top
    if(generate_top){
        sift(top,descriptors,features);
        CLOG(LTRACE)<<"SIFT top " << features.features.size() <<endl;
        for(int i=0; i < features.features.size(); i++){
            PointXYZSIFT point;
            int u = round(features.features[i].pt.x);
            int v = round(features.features[i].pt.y);
            if (mask_top && top_mask.at<float>(v, u)==0) {
                    continue;
            }

            int xx = (u)*(width)/(top.cols);
            int yy = (v)*(depth)/(top.rows);

            point.x = float(width-xx)/1000;
            point.y = float(-depth+yy)/1000;
            point.z = float(height)/1000;
            for(int j=0; j<descriptors.cols;j++){
                point.descriptor[j] = descriptors.row(i).at<float>(j);
            }
            point.multiplicity = 1;
            cloud_xyzsift->push_back(point);
        }
    }
    //bottom
        if(generate_bottom){
        sift(bottom,descriptors,features);
        CLOG(LTRACE)<<"SIFT bottom " << features.features.size() <<endl;
        for(int i=0; i < features.features.size(); i++){
            PointXYZSIFT point;
            int u = round(features.features[i].pt.x);
            int v = round(features.features[i].pt.y);
            if (mask_bottom && bottom_mask.at<float>(v, u)==0) {
                    continue;
            }

            int xx = (u)*(width)/(bottom.cols);
            int yy = (v)*(depth)/(bottom.rows);

            point.x = float(xx)/1000;
            point.y = float(-depth+yy)/1000;
            point.z = float(0)/1000;
            for(int j=0; j<descriptors.cols;j++){
                point.descriptor[j] = descriptors.row(i).at<float>(j);
            }
            point.multiplicity = 1;
            cloud_xyzsift->push_back(point);
        }
    }
    //left
    if(generate_left){
        sift(left,descriptors,features);
        CLOG(LTRACE)<<"SIFT left " << features.features.size() <<endl;
        for(int i=0; i < features.features.size(); i++){
            PointXYZSIFT point;
            int u = round(features.features[i].pt.x);
            int v = round(features.features[i].pt.y);
            if (mask_left && left_mask.at<float>(v, u)==0) {
                    continue;
            }

            int yy = (u)*(depth)/(left.cols);
            int zz = (v)*(height)/(left.rows);

            point.x = float(width)/1000;
            point.y = float(-depth+yy)/1000;
            point.z = float(height-zz)/1000;
            for(int j=0; j<descriptors.cols;j++){
                point.descriptor[j] = descriptors.row(i).at<float>(j);
            }
            point.multiplicity = 1;
            cloud_xyzsift->push_back(point);
        }
    }
    //right
    if(generate_right){
        sift(right,descriptors,features);
        CLOG(LTRACE)<<"SIFT right " << features.features.size() <<endl;
        for(int i=0; i < features.features.size(); i++){
            PointXYZSIFT point;
            int u = round(features.features[i].pt.x);
            int v = round(features.features[i].pt.y);

            int yy = (u)*(depth)/(right.cols);
            int zz = (v)*(height)/(right.rows);
            if (mask_right && right_mask.at<float>(v, u)==0) {
                    continue;
            }

            point.x = float(0)/1000;
            point.y = float(-yy)/1000;
            point.z = float(height-zz)/1000;
            for(int j=0; j<descriptors.cols;j++){
                point.descriptor[j] = descriptors.row(i).at<float>(j);
            }
            point.multiplicity = 1;
            cloud_xyzsift->push_back(point);
        }
    }

#endif
}



} //: namespace CuboidModelGenerator
} //: namespace Processors

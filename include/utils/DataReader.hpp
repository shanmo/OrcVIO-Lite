/*
 * @Descripttion: This header include functions and types for reading IMU and image data, and methods to manipulate.
 * @Author: Xiaochen Qiu
 */


#ifndef DATA_READER_H
#define DATA_READER_H


#include <iostream>
#include <fstream>
#include <vector>

#include "sensors/ImuData.hpp"

using std::cerr;
using std::string;
using std::vector;
using std::ifstream;
using std::getline;
using std::make_pair;
using std::pair;
using std::endl;

namespace orcvio {

struct ImgInfo {
	double timeStampToSec;
	std::string imgName;
};

/**
 * @description: Read data.csv containing image file names
 * @param imagePath Path of data.csv
 * @return: iListData Cotaining image informations.
 */
void loadImageList(const std::string imagePath, std::vector<ImgInfo> &iListData) {
    std::ifstream inf;
    inf.open(imagePath, std::ifstream::in);
    const int cnt = 2;         
    std::string line;
    int j = 0;
    size_t comma = 0;
    size_t comma2 = 0;
    ImgInfo temp;

    std::getline(inf,line);	
    while (!inf.eof()) {
        std::getline(inf,line);

        comma = line.find(',',0);	
		temp.timeStampToSec = 1e-9*atol(line.substr(0,comma).c_str());		
        
        while (comma < line.size() && j != cnt-1) {
            comma2 = line.find(',',comma + 1);	
            temp.imgName = line.substr(comma + 1,comma2-comma-1).c_str();
            ++j;
            comma = comma2;
        }

        iListData.push_back(temp);
        j = 0;
    }

    inf.close();
}

/**
 * @description: Read data.csv containing IMU data
 * @param imuPath Path of data.csv
 * @return: vimuData Cotaining IMU informations.
 */
void loadImuFile(const std::string imuPath, std::vector<ImuData> &vimuData) {
    std::ifstream inf;
    inf.open(imuPath, std::ifstream::in);
    const int cnt = 7;        
    std::string line;
    int j = 0;
    size_t comma = 0;
    size_t comma2 = 0;
    char imuTime[14] = {0};
    double acc[3] = {0.0};
    double grad[3] = {0.0};
    double imuTimeStamp = 0;

    std::getline(inf,line);		
    while (!inf.eof()) {
        std::getline(inf,line);

        comma = line.find(',',0);
		std::string temp = line.substr(0,comma);
		imuTimeStamp = 1e-9*atol(line.substr(0,comma).c_str());	
        
        while (comma < line.size() && j != cnt-1) {
            comma2 = line.find(',',comma + 1);
            switch(j) {
			case 0:
				grad[0] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 1:
				grad[1] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 2:
				grad[2] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 3:
				acc[0] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 4:
				acc[1] = atof(line.substr(comma + 1,comma2-comma-1).c_str());
				break;
			case 5:
				acc[2] = atof(line.substr(comma + 1,comma2-comma-1).c_str());	
				break;
            }
            ++j;
            comma = comma2;
        }
		ImuData tempImu(imuTimeStamp, grad[0], grad[1], grad[2], acc[0], acc[1], acc[2]);
        vimuData.push_back(tempImu);
        j = 0;
    }

	inf.close();
}


template<typename T>
T parse_next(std::stringstream& liness, const char delimiter) {
    std::string ele;
    std::getline(liness, ele, delimiter);
    std::stringstream eless(ele);
    T value;
    if (! (eless >> value)) {
        throw new std::runtime_error("unable to parse: " + ele);
    }
    return value;
}

/**
 * @description: Read data.csv containing IMU data
 * @param imuPath Path of data.csv
 * @return: vimuData Cotaining IMU informations.
 */


template <typename T>
using eigen_vector = std::vector<T, Eigen::aligned_allocator<T> >;

void loadGTFile(const std::string gt_poses_file,
                std::vector<double>& gt_pose_timestamps,
                eigen_vector<Eigen::Isometry3d> &allGTPoses) {
    char delimiter = ' ';
    if (gt_poses_file.substr(gt_poses_file.size()-4, 4) == ".csv") {
        delimiter = ',';
    }
    std::ifstream infile;
    infile.open(gt_poses_file, std::ifstream::in);

    if (!infile) {
        std::cerr << "file : '" << gt_poses_file << "' failed! \n";
        throw new std::runtime_error("Unable to open file");
    }

    std::string line;
    bool is_first_pose = true;
    Eigen::Isometry3d first_transform;

    while (std::getline(infile, line)) {
        if (line[0] == '#')
            continue;
        std::stringstream liness(line);
        double timestamp;
        if (delimiter == ',') {
            timestamp = parse_next<uint64_t>(liness, delimiter) / 1e9;
        } else {
            timestamp = parse_next<double>(liness, delimiter);
        }

        gt_pose_timestamps.push_back(timestamp);
        Eigen::Translation3d pos;
        pos.x() = parse_next<double>(liness, delimiter);
        pos.y() = parse_next<double>(liness, delimiter);
        pos.z() = parse_next<double>(liness, delimiter);

        double qx = parse_next<double>(liness, delimiter);
        double qy = parse_next<double>(liness, delimiter);
        double qz = parse_next<double>(liness, delimiter);
        double qw = parse_next<double>(liness, delimiter);
        // std::cout << "line: " << timestamp << ", " << pos.x() << ", " << pos.y()
        //           << ", " << pos.z() << ", " << qx << ", " << qy << ", " << qz << "\n";
        Eigen::Quaterniond quat(qw, qx, qy, qz);
        Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
        transform.rotate(quat);
        transform *= pos;
        if (is_first_pose) {
            first_transform = transform;
            is_first_pose = false;
        } else {
            transform = first_transform.inverse() * transform;
        }
        allGTPoses.push_back(transform);
        std::cout << "ts: " << timestamp << ", pos:" << transform.translation() << "\n";
    }
	infile.close();
}


bool findFirstAlign(const std::vector<ImuData> &vImu, const std::vector<ImgInfo> &vImg, std::pair<int,int> &ImgImuAlign) {
	double imuTime0 = vImu[0].timeStampToSec;
	double imgTime0 = vImg[0].timeStampToSec;
	
	if(imuTime0>imgTime0) {		
		for(size_t i=1; i<vImg.size(); i++) {
			double imgTime = vImg[i].timeStampToSec;
			if(imuTime0<=imgTime) {
				for(size_t j=0; j<vImu.size(); j++) {
					double imuTime = vImu[j].timeStampToSec;
					if(imuTime==imgTime) {
						int imgID = i;
						int imuID = j;
						ImgImuAlign = std::make_pair(imgID,imuID);
						return true;
					}
				}
				return false;
			}
		}
		return false;
	}
	else if(imuTime0<imgTime0) {	
		for(size_t i=1; i<vImu.size(); i++) {
			double imuTime = vImu[i].timeStampToSec;
			if(imuTime==imgTime0) {
				int imgID = 0;
				int imuID = i;
				ImgImuAlign = std::make_pair(imgID,imuID);
				return true;
			}
		}
		return false;
	}
	else {						
		int imgID = 0;
		int imuID = 0;
		ImgImuAlign = std::make_pair(imgID,imuID);
		return true;
	}
}

}


#endif // DATA_READER_H

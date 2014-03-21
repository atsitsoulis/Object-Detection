#ifndef UTILITY_FUNCTIONS_HPP
#define	UTILITY_FUNCTIONS_HPP

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

void mouseHandler(
    int event, 
    int x, 
    int y, 
    int z, 
    void* image_roi
);

void extractFeaturesAndDescriptors(
    Mat &frame,
    vector< KeyPoint > &frame_keypoints,
    Mat &frame_descriptors
);

void detectObject(
    vector< DMatch > &good_matches,
    vector< KeyPoint > &frame_keypoints,
    vector< KeyPoint > &image_roi_keypoints,
    Mat &image_roi,
    Mat &img_matches
);

#endif	/* UTILITY_FUNCTIONS_HPP */


#include "utility_functions.hpp"

// -----------------------------------------------------------------------------
// GLOBAL VARIABLES
// -----------------------------------------------------------------------------
// Flags updated according to left mouse button activity
extern bool ldown; 
extern bool lup;
// Original image
extern Mat frame;
extern Mat target_frame;
// Starting and ending points of the user's selection
extern Point corner1;
extern Point corner2;
// ROI
extern Rect bbox;
// Window names
extern string window_main;
extern string window_cropping;
extern string window_result;

// -----------------------------------------------------------------------------
// CLASSES AND FUNCTIONS
// -----------------------------------------------------------------------------
// Mouse handler for selection of ROI from user's mouse input (rectangle)
void mouseHandler(
    int event, 
    int x, 
    int y, 
    int z, 
    void* image_roi
) {
	// When the left mouse button is pressed, record its position and save it in 
    // corner1 
	if(event == EVENT_LBUTTONDOWN) {	
		ldown = true;
		corner1.x = x;
		corner1.y = y;
	}
	
	// When the left mouse button is released, record its position and save it 
    // in corner2 
	if(event == EVENT_LBUTTONUP) {	
		// Also check if user selection is bigger than 20 pixels
		if( (abs(x - corner1.x) > 20) && 
            (abs(y - corner1.y) > 20)
        ) {
			lup = true;
			corner2.x = x;
			corner2.y = y;
		} else {
			cout << "Please select a bigger region" << endl;
			ldown = false;
		}
	}

	// Update the bbox showing the selected region as the user drags the mouse
	if( 
        (ldown == true) && 
        (lup == false) 
    ) {
		Point pt;
		pt.x = x;
		pt.y = y;
		Mat local_img = target_frame.clone();
		rectangle(
            local_img, 
            corner1, 
            pt, 
            Scalar(
                0, 
                0, 
                255
            )
        );
		imshow(
            window_cropping, 
            local_img
        );
	}
	
	// Define ROI and crop it out when both corners have been selected	
	if( 
        (ldown == true) && 
        (lup == true) 
    ) {
		bbox.width = abs(corner1.x - corner2.x);
		bbox.height = abs(corner1.y - corner2.y);
		bbox.x = min(corner1.x, corner2.x);
		bbox.y = min(corner1.y, corner2.y);

		// Make an image out of just the selected ROI and display it in a new 
        // window
        
        target_frame(bbox).copyTo(* (Mat *)image_roi);
                        		
		ldown = false;
		lup = false;
	}
}

// Extraction of features and descriptors
void extractFeaturesAndDescriptors(
    Mat &frame,
    vector< KeyPoint > &frame_keypoints,
    Mat &frame_descriptors
) {
    Mat frame_gray;
    // Detect the keypoints using SURF Detector
	int minHessian = 300;
	SurfFeatureDetector detector( minHessian );
	SurfDescriptorExtractor extractor;
    
    // Convert REFERENCE to gray
    cvtColor(
        frame,
        frame_gray,
        CV_BGR2GRAY
    );

    // Detect keypoints in REFERENCE
    detector.detect( 
        frame_gray, 
        frame_keypoints 
    );

    // Extract corresponding descriptors in REFERENCE
    extractor.compute( 
        frame_gray, 
        frame_keypoints, 
        frame_descriptors
    );
}

// Object detection and bounding box visualization
void detectObject(
    vector< DMatch > &good_matches,
    vector< KeyPoint > &frame_keypoints,
    vector< KeyPoint > &image_roi_keypoints,
    Mat &image_roi,
    Mat &img_matches
) {
     // Points for corresponding keypoints
    vector<Point2f> frame_points;
    vector<Point2f> image_roi_points;
    vector<Point2f> image_roi_corners_in_scene(4);
    vector<Point2f> image_roi_corners(4);
    // Transformation matrix (homography)
    Mat H;
            
    for( int i = 0; i < good_matches.size(); i++ ) {
        //Get the keypoints from the good matches
        image_roi_points.push_back( 
            image_roi_keypoints[ good_matches[i].queryIdx ].pt 
        );
        frame_points.push_back( 
            frame_keypoints[ good_matches[i].trainIdx ].pt 
        );
    }

    //Get the corners from the template
    image_roi_corners[0] = cvPoint(
        0,
        0
    );
    image_roi_corners[1] = cvPoint( 
        image_roi.cols, 
        0 
    );
    image_roi_corners[2] = cvPoint( 
        image_roi.cols, 
        image_roi.rows 
    );
    image_roi_corners[3] = cvPoint( 
        0, 
        image_roi.rows 
    );

    // Calculate transformation (homography) from correspondences
    H = findHomography( 
        image_roi_points, 
        frame_points, 
        CV_RANSAC 
    );

    // Transform points according to H
    perspectiveTransform( 
        image_roi_corners, 
        image_roi_corners_in_scene, 
        H
    );

    // Draw lines between the corners (the mapped object in the scene 
    // image )
    line( 
        img_matches, 
        image_roi_corners_in_scene[0] + 
            Point2f( 
                image_roi.cols, 
                0
            ), 
        image_roi_corners_in_scene[1] + 
            Point2f( 
                image_roi.cols, 
                0
            ), 
        Scalar(
            0, 
            255, 
            0
        ), 
        4 
    );
    line( 
        img_matches, 
        image_roi_corners_in_scene[1] + 
            Point2f( 
                image_roi.cols, 
                0
            ), 
        image_roi_corners_in_scene[2] + 
            Point2f( 
                image_roi.cols, 
                0
            ), 
        Scalar( 
            0, 
            255, 
            0
        ), 
        4 
    );
    line( 
        img_matches, 
        image_roi_corners_in_scene[2] + 
            Point2f( 
                image_roi.cols, 
                0
            ), 
        image_roi_corners_in_scene[3] + 
            Point2f( 
                image_roi.cols, 
                0
            ), 
        Scalar( 
            0, 
            255, 
            0
        ), 
        4 
    );
    line( 
        img_matches, 
        image_roi_corners_in_scene[3] + 
            Point2f( 
                image_roi.cols, 
                0
            ), 
        image_roi_corners_in_scene[0] + 
            Point2f( 
                image_roi.cols, 
                0
            ), 
        Scalar( 
            0, 
            255, 
            0
        ), 
        4 
    );
    
    String window_name;
    window_name = "Matching";
    imshow( 
        window_name, 
        img_matches 
    );
}    

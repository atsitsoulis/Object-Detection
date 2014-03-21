/*  Object matching using FLANN and SURF descriptors
 *  
 *  Algorithm:
 *  1)  Detection of SURF interest points and descriptors for every frame
 *      captures from the camera and a subimage containing the object of 
 *      interest (selected by the user). 
 *  2)  Using the k Nearest Neighbors matching algorithm, if more than 4 
 *      corresponding descriptors are found, the object is found.
 *  3)  The (homography) transformation matrix that transforms the subimage
 *      coordinates to those of the camera frame is calculated using the 
 *      corresponding points. 
 *  4)  Application of the homography reprojects the boundary of the object of
 *      interest to the scene and allows visualization.
 * 
 *  Usage:
 *      Press 's' in the main window (live video from camera) to take a 
 *      screenshot. In the new window, select the region of interest (ROI) by 
 *      enclosing with a rectangular bounding box, dragged with the mouse (left 
 *      button). The ROI should contain distinct features for better results 
 *      (e.g. tests using a book cover produced remarkable results). The new 
 *      window shows the results. If the ROI contains a movable objects, results
 *      will show that the detector is invariant of scale and rotations. 'Esc'
 *      terminates the program.
*/

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>
#include <vector>
#include "utility_functions.hpp"

using namespace std;
using namespace cv;

// -----------------------------------------------------------------------------
// GLOBAL VARIABLES
// -----------------------------------------------------------------------------
// Flags updated according to left mouse button activity
bool ldown = false; 
bool lup = false;
// Original image
Mat frame;
Mat target_frame;
// Starting and ending points of the user's selection
Point corner1;
Point corner2;
// ROI
Rect bbox;
// Window names
string window_main = "Main window";
string window_cropping = "Select ROI";
string window_result = "Result";

//------------------------------------------------------------------------------
// MAIN FUNCTION
//------------------------------------------------------------------------------
int main(
    int argc,
    char** argv
) {
	FlannBasedMatcher matcher;

    // Detection in live camera input
    VideoCapture capture(-1);
    if( ! capture.isOpened() ) {
		return -1;
	}
    
    Mat *image_roi = new Mat;
    
    // Press 'Esc' to terminate
    while (waitKey(1) != 27) {
        capture >> frame;
		
        imshow( 
            window_main, 
            frame 
        );
                        
        // Press 's' to freeze capture and crop target
        if( waitKey(1) == 115 ) {
            target_frame = frame.clone();

            // Create window for frozen frame
            imshow( 
                window_cropping, 
                target_frame 
            );
            moveWindow(
                window_cropping,
                200,
                200
            );

            // Setup the mouse callback. 
            setMouseCallback(
                window_cropping, 
                mouseHandler, 
                (void *) image_roi
            );
        }

        // If ROI has been selected, proceed with matching
        if( 
            (image_roi->cols > 0) &&
            (image_roi->rows > 0)
        ) {
            vector< KeyPoint > frame_keypoints;
            Mat frame_descriptors;
            vector< KeyPoint > image_roi_keypoints;
            Mat image_roi_descriptors;
            
            // Detect keypoints and extract descriptors
            extractFeaturesAndDescriptors(
                frame,
                frame_keypoints,
                frame_descriptors
            );
            
            extractFeaturesAndDescriptors(
                *image_roi,
                image_roi_keypoints,
                image_roi_descriptors
            );
            
            // Matching descriptor vectors using FLANN matcher
            vector< vector< DMatch > > matches;
            matcher.knnMatch(
                image_roi_descriptors, 
                frame_descriptors,
                matches, 
                2
            );

            // Thresholding to filter out the "bad" matches
            vector< DMatch > good_matches;
            for( int j = 0; j < image_roi_descriptors.rows; j++ ) { 
                if(
                    (matches[j][0].distance <= 0.6*(matches[j][1].distance)) && 
                    ((int) matches[j].size() <= 2 ) && 
                    ((int) matches[j].size() > 0)
                ) {
                    good_matches.push_back(matches[j][0]);
                }
            }

            // Draw only "good" matches
            Mat img_matches;
            drawMatches( 
                *image_roi, 
				image_roi_keypoints, 
                frame, 
                frame_keypoints,
                good_matches, 
                img_matches, 
                Scalar::all(-1), 
                Scalar::all(-1),
                vector<char>(), 
                DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS 
            );

            // If more than 4 keypoint correspondences are found, object is
            // detected
            if (good_matches.size() >= 4) {
                // Object detection and bounding box visualization
                detectObject(
                    good_matches,
                    frame_keypoints,
                    image_roi_keypoints,
                    *image_roi,
                    img_matches
                );
            }

            imshow( 
                "Main window", 
                frame 
            );
        }
            
    }
    
    delete image_roi;
    image_roi = NULL;
    
	return 0;
}


Object-Detection
================

Object matching using FLANN and SURF descriptors

Algorithm:
1)  Detection of SURF interest points and descriptors for every frame
    captures from the camera and a subimage containing the object of 
    interest (selected by the user). 
2)  Using the k Nearest Neighbors matching algorithm, if more than 4 
    corresponding descriptors are found, the object is found.
3)  The (homography) transformation matrix that transforms the subimage
    coordinates to those of the camera frame is calculated using the 
    corresponding points. 
4)  Application of the homography reprojects the boundary of the object of
    interest to the scene and allows visualization.

Usage:
    Press 's' in the main window (live video from camera) to take a 
    screenshot. In the new window, select the region of interest (ROI) by 
    enclosing with a rectangular bounding box, dragged with the mouse (left 
    button). The ROI should contain distinct features for better results 
    (e.g. tests using a book cover produced remarkable results). The new 
    window shows the results. If the ROI contains a movable objects, results
    will show that the detector is invariant of scale and rotations. 'Esc'
    terminates the program.

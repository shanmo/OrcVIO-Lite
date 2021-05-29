#message for triggering object detection of a specific object
#contains object type to trigger corresponding node and region of interest to constrain search
Header header
string obj_type #object type, reference to the object in the database
#geometry_msgs/Pose coarse_pose #coarse pose from 2d detector
wm_od_interface_msgs/ROI2d region2d #image region of interest used to constrain object detection

## about  

* this repo implements bbox tracking using sort
* original version uses IOU, but if the overlap is 0, cannot re-identify the same object and lead to ID switch 
* this version add centroid distance, so that we could use a threshold to determine whether the two centroids belong to the same object 

## dependencies 

- opencv
- [catkin simple](https://github.com/catkin/catkin_simple)
- [darknet ros](https://github.com/leggedrobotics/darknet_ros)
- [darknet ros bridge](https://github.com/moshanATucsd/OrcVIO_Lite/tree/master/ros_wrapper/src/darknet_ros_bridge)
- [wm_od_interface_msgs](https://gitlab.sitcore.net/aimm/phoenix-r1/-/tree/master/src/common_msgs/wm_od_interface_msgs)

## notes 

* we only track valid class, e.g. barrel, which can be set in `gen_class` function 
* the `centroid_dist_threshold` determines whether we consider two centroids belong to same object, which can be tuned in launch file. set to a large value if we only need to detect barrels to accommodate jackal's aggressive motion 
* the output topic is `/SortRos/tracked_bbox`, here is the output from `rostopic echo /SortRos/tracked_bbox`

```
header: 
  seq: 49
  stamp: 
    secs: 1317067372
    nsecs: 169100046
  frame_id: "tracked bboxes"
bounding_boxes: 
  - 
    xmin: 433
    ymin: 164
    xmax: 455
    ymax: 453
    id: 18
    Class: "car"
    lost_flag: False
  - 
    xmin: 19
    ymin: 159
    xmax: 69
    ymax: 51
    id: 21
    Class: "car"
    lost_flag: False
  - 
    xmin: 117
    ymin: 159
    xmax: 181
    ymax: 150
    id: 23
    Class: "car"
    lost_flag: False
  - 
    xmin: 313
    ymin: 171
    xmax: 347
    ymax: 338
    id: 25
    Class: "car"
    lost_flag: True
  - 
    xmin: 201
    ymin: 162
    xmax: 261
    ymax: 232
    id: 27
    Class: "car"
    lost_flag: False
  - 
    xmin: 538
    ymin: 166
    xmax: 552
    ymax: 550
    id: 28
    Class: "car"
    lost_flag: False
  - 
    xmin: 535
    ymin: 171
    xmax: 548
    ymax: 546
    id: 29
    Class: "car"
    lost_flag: False
```

## references 

- https://github.com/tryolabs/norfair
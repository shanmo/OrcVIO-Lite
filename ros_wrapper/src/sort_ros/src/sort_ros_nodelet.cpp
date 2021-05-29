#include "nodelet/nodelet.h"
#include <pluginlib/class_list_macros.h>
#include "std_msgs/String.h"
#include "sensor_msgs/Image.h"

#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <darknet_ros_msgs/BoundingBoxes.h>
#include "sort_ros/sort_tracking.h"
#include "sort_ros/TrackedBoundingBoxes.h"
#include "sort_ros/TrackedBoundingBox.h"

using namespace std;


// namespace example_pkg
namespace sort_ros
{

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// class MyNodeletClass : public nodelet::Nodelet
class SortRos : public nodelet::Nodelet
{

private:

virtual void onInit()
{

    auto private_nh = getPrivateNodeHandle();
    auto nh = getNodeHandle();
    NODELET_WARN("Initialized the SORT Nodelet");

    std::string image_topic, bbox_topic, tracked_bbox_topic, visualization_topic;

    // we load this from the param defined in launch file, default is the detection image from darknet ros 
    private_nh.param<std::string>("image_topic", image_topic,  "darknet_ros/detection_image");
    ROS_INFO_STREAM("image_topic: " << image_topic);
    
    private_nh.param<std::string>("bbox_topic", bbox_topic,  "darknet_ros/bounding_boxes");
    private_nh.param<std::string>("tracked_bbox_topic", tracked_bbox_topic, "tracked_bbox");
    private_nh.param<std::string>("visualization_topic", visualization_topic, "detection_image");

    // load params
    Config config;
    private_nh.param<int>("max_age", config.max_age,  3);
    private_nh.param<int>("min_hits", config.min_hits, 5);
    private_nh.param<double>("iou_threshold", config.iou_threshold, 0.3);
    private_nh.param<double>("centroid_dist_threshold", config.centroid_dist_threshold, 50);
    private_nh.param<bool>("use_centroid_dist_flag", config.use_centroid_dist_flag, true);
    sort_tracker.set_config(config);

    boundingBoxesPublisher_ = private_nh.advertise<sort_ros::TrackedBoundingBoxes>(tracked_bbox_topic, 10);

    // detectionImagePublisher_ = private_nh.advertise<sensor_msgs::Image>(visualization_topic, 1, true);
    image_trans_ = make_unique<image_transport::ImageTransport>(private_nh);
    detectionImagePublisher_ = image_trans_->advertise(visualization_topic, 10);
    
    namespace sph = std::placeholders; // for _1, _2, ...
    image_sub_ = make_unique<message_filters::Subscriber<sensor_msgs::Image>>(nh, image_topic, 1);
    bbox_sub_ = make_unique<message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>>(nh, bbox_topic, 1);


    // sub_ = make_unique<message_filters::TimeSynchronizer<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes>>(*image_sub_, *bbox_sub_, 10);
    // ApproximateTime takes a queue size as its constructor argument, hence MySyncPolicy(10)
    sub_ = make_unique<message_filters::Synchronizer<MySyncPolicy> > (MySyncPolicy(100), *image_sub_, *bbox_sub_);

    sub_->registerCallback(std::bind(&SortRos::messageCb, this, sph::_1, sph::_2));

}

// must use a ConstPtr callback to use zero-copy transport
void messageCb(const sensor_msgs::ImageConstPtr& message,
                const darknet_ros_msgs::BoundingBoxesConstPtr& bboxes) {

    // track bbox 
    vector<TrackingBox> detFrameData;
    for (auto bbox : bboxes->bounding_boxes)
    {

        TrackingBox tb;
        tb.box = Rect_<float>(Point_<float>(bbox.xmin, bbox.ymin), Point_<float>(bbox.xmax, bbox.ymax));
        tb.object_class = bbox.Class;

        // if (sort_tracker.check_valid_class(bbox.Class)) 
        if (sort_tracker.check_valid_class(bbox.Class) && (bbox.probability > 0.5)) 
            detFrameData.push_back(tb);
    }

    sort_tracker.update(detFrameData);

    // display tracking results 
    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(message, "bgr8");
    detectionImage = sort_tracker.draw_bbox(img->image);
    detectionImage = sort_tracker.draw_centroids(detectionImage);

    boundingBoxesResults_.header.stamp = img->header.stamp;

    trackedBboxPublisher();
    publishDetectionImage();
}

void trackedBboxPublisher()
{
    if (boundingBoxesPublisher_.getNumSubscribers() < 1)
        return;

    boundingBoxesResults_.bounding_boxes.clear();

    for (auto it = sort_tracker.trackers.begin(); it != sort_tracker.trackers.end(); it++)
    {

        sort_ros::TrackedBoundingBox boundingBox;

        boundingBox.Class = (*it).object_class;
        boundingBox.id = (*it).m_id + 1;
        boundingBox.lost_flag = (*it).lost_flag;

        Rect_<float> box = (*it).get_state();
        boundingBox.xmin = box.x;
        boundingBox.ymin = box.y;
        boundingBox.xmax = box.x + box.width;
        boundingBox.ymax = box.y + box.height;
        boundingBoxesResults_.bounding_boxes.push_back(boundingBox);

    }

    boundingBoxesResults_.header.frame_id = "tracked bboxes";
    boundingBoxesPublisher_.publish(boundingBoxesResults_);
}

void publishDetectionImage()
{
    if (detectionImagePublisher_.getNumSubscribers() < 1)
        return;

    cv_bridge::CvImage cvImage;
    cvImage.header.stamp = ros::Time::now();
    cvImage.header.frame_id = "detection_image";
    cvImage.encoding = sensor_msgs::image_encodings::BGR8;
    cvImage.image = detectionImage;
    
    detectionImagePublisher_.publish(*cvImage.toImageMsg());
    
    // ROS_DEBUG("Detection image has been published.");

}

std::unique_ptr<message_filters::Subscriber<sensor_msgs::Image>> image_sub_;
std::unique_ptr<message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>> bbox_sub_;

typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes> MySyncPolicy;
// std::unique_ptr<message_filters::TimeSynchronizer<sensor_msgs::Image, darknet_ros_msgs::BoundingBoxes>> sub_;
std::unique_ptr<message_filters::Synchronizer<MySyncPolicy> > sub_;

ros::Publisher boundingBoxesPublisher_;

// ros::Publisher detectionImagePublisher_;
std::unique_ptr<image_transport::ImageTransport> image_trans_;
image_transport::Publisher detectionImagePublisher_;

sort_ros::SortTracker sort_tracker;
sort_ros::TrackedBoundingBoxes boundingBoxesResults_;

Mat detectionImage;

};

}

// watch the capitalization carefully
// PLUGINLIB_EXPORT_CLASS(example_pkg::MyNodeletClass, nodelet::Nodelet)
PLUGINLIB_EXPORT_CLASS(sort_ros::SortRos, nodelet::Nodelet);

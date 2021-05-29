#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#include "boost/format.hpp"
#include "boost/filesystem.hpp"

#include "orcvio/obj/ObjectFeatureInitializer.h"
#include "orcvio/obj/ObjectFeature.h"
#include "orcvio/feat/FeatureInitializer.h"
#include "orcvio/tests/test_utils.h"
#include "orcvio/utils/se3_ops.hpp"

using orcvio::dsread;
using orcvio::ObjectFeature;
using orcvio::FeatureInitializer;
using boost::filesystem::exists;
using boost::filesystem::path;
namespace fs = boost::filesystem;


namespace orcvio {

constexpr bool DEBUG = false;

void visualize_multiframe_test_data(const std::shared_ptr<ObjectFeature>& obj_obs_ptr,
                                    const Eigen::MatrixX3d& kps_gt_3d_world,
                                    const std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>>& clones_cam,
                                    const Eigen::Matrix4d& wTq)
{
  // intrinsic matrix
  Eigen::Matrix3d K;
  K << 200., 0, 320.,
    0, 200., 240.,
    0, 0, 1;
  cv::Scalar obs_color{0, 0, 255};
  cv::Scalar gt_color{0, 255, 0};

  for (size_t t = 0; t < obj_obs_ptr->timestamps[0].size(); ++t) {
    auto const& zs_per_frame = obj_obs_ptr->zs[t];
    cv::Mat frame(K(1,2)*2, K(0,2)*2, CV_8UC3, cv::Scalar(0, 0, 0));

    // Plot observed keypoints
    for (int r = 0; r < zs_per_frame.rows(); ++r) {
      if (zs_per_frame.allFinite()) {
        Eigen::Vector2i pt = (K.topLeftCorner<2, 2>() * zs_per_frame.row(r).transpose() + K.topRightCorner<2, 1>()).cast<int>();
        cv::drawMarker(frame, {pt(0), pt(1)}, obs_color);
      }
    }

    // Plot gt keypoints
    FeatureInitializer::ClonePose clonecam = clones_cam.at(0).at(t);
    for (int r = 0; r < kps_gt_3d_world.rows(); ++r) {
        Eigen::Vector3d kps_gt_3d_cam = clonecam.transformGlobalToCam(kps_gt_3d_world.row(r).transpose());
        Eigen::Vector2d kps_gt_2d = kps_gt_3d_cam.head<2>() / kps_gt_3d_cam(2);
        Eigen::Vector2i pt = (K.topLeftCorner<2, 2>() * kps_gt_2d + K.topRightCorner<2, 1>()).cast<int>();
        cv::drawMarker(frame, {pt(0), pt(1)}, gt_color);
    }


    // Plot observed bounding boxes
    if (obj_obs_ptr->zb.size() > t) {
      auto const& zb_per_frame = obj_obs_ptr->zb[t];
      Eigen::Vector2i topleft = (K.topLeftCorner<2, 2>() * zb_per_frame.head<2>() + K.topRightCorner<2, 1>()).cast<int>();
      Eigen::Vector2i bottomright = (K.topLeftCorner<2, 2>() * zb_per_frame.tail<2>() + K.topRightCorner<2, 1>()).cast<int>();
      cv::rectangle(frame, {topleft(0), topleft(1)}, {bottomright(0), bottomright(1)}, obs_color);
    }

    cv::imshow("c", frame);
    cv::waitKey(30);
  }
}

int load_multi_frame_test_data(const std::string& filename_fmt,
                               std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>>& clones_cam,
                               std::shared_ptr<ObjectFeature>& obj_obs_ptr,
                               Eigen::Matrix4d& wTq_gt,
                               Eigen::Vector3d& object_mean_shape,
                               Eigen::MatrixX3d& object_keypoints_mean)
{
  boost::format filename_fmt_b(filename_fmt);
  size_t cam_id = 0;
  std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;
  Eigen::MatrixX3d kps_gt_3d;
  for (int i = 0; i < 100 && exists(path((filename_fmt_b % i).str())); i++)
  {

    // insert timestamps into object observations
    obj_obs_ptr->timestamps[cam_id].emplace_back(i);

    // construct filename
    std::string filename = (filename_fmt_b % i).str();

    // load test data
    auto name2mat = cv::hdf::open(filename);

    // zs : groundtruth semantic measurements 
    auto const zs_per_frame = dsread(name2mat, "zs");
    obj_obs_ptr->zs.push_back(zs_per_frame);

    // zb : groundtruth bbox measurements in x,y,width, height format
    if ( name2mat->hlexists("zb") ) {
      Eigen::Vector4d xywh = dsread(name2mat, "zb").transpose();
      Eigen::Vector4d xyminmax;
      xyminmax << xywh(0), xywh(1), xywh(0) + xywh(2), xywh(1) + xywh(3);
      obj_obs_ptr->zb.push_back(xyminmax);
    }

    // wTo : groundtruth optical frame to World transform (4 x 4)
    auto const wTo = dsread(name2mat, "wTo");

    clones_cami.insert({i, FeatureInitializer::ClonePose(wTo)});

    // wTq : groundtruth object to World transform (4 x 4)
    wTq_gt = dsread(name2mat, "wTq");

    // 12 x 3 groundtruth keypoints coordinates
    kps_gt_3d = dsread(name2mat, "kps_gt_3d");

    // 12 x 3 mean shape keypoints coordinates
    object_keypoints_mean = dsread(name2mat, "mean_shape");

    // 3 x 1 ellipsoid shape
    object_mean_shape = dsread(name2mat, "ellipsoid_shape");
  }

  clones_cam.insert({cam_id, clones_cami});

  if (DEBUG)
    visualize_multiframe_test_data(obj_obs_ptr, kps_gt_3d, clones_cam, wTq_gt);
    
  return 1;
}


const std::string ensure_path_exists(const std::string& filepath)
{
    if ( ! fs::exists(fs::path(filepath))) {
        throw std::runtime_error("filepath (" + filepath + ") not found in working directory: " + fs::current_path().string() + ". Please run from the top of the orcvio_cpp project directory.");
    }
    return filepath;
}


} // namespace orcvio

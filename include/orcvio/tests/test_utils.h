#ifndef TEST_UTILS_H
#define TEST_UTILS_H 1

#include <unordered_map>
#include <memory>
#include <boost/format.hpp>
#include <Eigen/Dense>

#include "orcvio/obj/ObjectFeatureInitializer.h"
#include "orcvio/obj/ObjectFeature.h"
#include "orcvio/feat/FeatureInitializer.h"


namespace orcvio {
  int load_multi_frame_test_data(const std::string& filename,
                                 std::unordered_map<size_t, std::unordered_map<double, orcvio::FeatureInitializer::ClonePose>>& clones_cam,
                                 std::shared_ptr<orcvio::ObjectFeature>& obj_obs_ptr,
                                 Eigen::Matrix4d& wTq_gt,
                                 Eigen::Vector3d& object_mean_shape,
                                 Eigen::MatrixX3d& object_keypoints_mean);
  const std::string ensure_path_exists(const std::string& filepath);
} // namespace orcvio

#endif // TEST_UTILS_H

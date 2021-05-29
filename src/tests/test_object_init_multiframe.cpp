#include <memory>
#include <boost/format.hpp>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>

#include "orcvio/obj/ObjectFeature.h"
#include "orcvio/obj/ObjectFeatureInitializer.h"
#include "orcvio/utils/se3_ops.hpp"
#include "orcvio/tests/test_utils.h"

using namespace std;
using namespace Eigen;

using orcvio::vector_eigen;

using namespace orcvio;
using orcvio::load_multi_frame_test_data;
using orcvio::ensure_path_exists;
using boost::format;

// TEST(TestSuite, testCase1)
TEST(ObjectInitMultiFrame, test_object_init_multiframe) 
{

  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;

  // initialize object feature structure
  std::shared_ptr<ObjectFeature> obj_obs_ptr;
  obj_obs_ptr.reset(new ObjectFeature(1, "car"));

  Eigen::Matrix4d wTq_gt;
  Eigen::Vector3d object_mean_shape;
  Eigen::MatrixX3d object_keypoints_mean;

  std::unique_ptr<ObjectFeatureInitializer> object_feat_init;
  Matrix3d camera_intrinsics = Matrix3d::Identity();
  FeatureInitializerOptions featinit_options;

  std::vector<string> input_file_names{"src/tests/data/one_car_no_zb/frame_%d.h5",
                                       "src/tests/data/one_car/frame_%d.h5"};

  for (auto const & input_file_name : input_file_names) {

      ensure_path_exists((format(input_file_name) % 0).str());

      if (clones_cam.size())
          clones_cam.erase(clones_cam.begin());

      obj_obs_ptr.reset(new ObjectFeature(1, "car"));
      load_multi_frame_test_data(input_file_name,
                                  clones_cam,
                                  obj_obs_ptr,
                                  wTq_gt,
                                  object_mean_shape,
                                  object_keypoints_mean);

      std::cout << "total frame no.: " << obj_obs_ptr->timestamps[0].size() << std::endl; 

      object_feat_init.reset(
          new ObjectFeatureInitializer(featinit_options,
                                        object_mean_shape,
                                        object_keypoints_mean,
                                        camera_intrinsics));

      bool init_success_flag;
      Eigen::Matrix4d wTq_est;
      std::tie(init_success_flag, wTq_est) =
      object_feat_init->single_object_initialization(obj_obs_ptr,
                                                      clones_cam);

      //ASSERT_NEAR((wTq_gt - wTq_est).norm(), 0, 6e-1)
      //  << "wTq_est: \n" << wTq_est << "\n" << "wTq_gt: \n" << wTq_gt << "\n";
      double dispR, dispt;
      std::tie(dispR, dispt) = displacement(wTq_gt, wTq_est);
      EXPECT_NEAR(dispR, 0, 0.5)
      << "Rotation does not match. "
      << "\nwTq_est: \n" << wTq_est << "\n" << "wTq_gt: \n" << wTq_gt << "\n";
      ASSERT_NEAR(dispt, 0, 3.5e-1)
      << "translation does not match. "
      << "\nwTq_est: \n" << wTq_est << "\n" << "wTq_gt: \n" << wTq_gt << "\n";
  }


  /*
  load_multi_frame_test_data("src/tests/data/one_car/frame_%d.h5",
                             clones_cam,
                             obj_obs_ptr,
                             wTq_gt,
                             object_mean_shape,
                             object_keypoints_mean);

  object_feat_init.reset(new ObjectFeatureInitializer(featinit_options,
                                                      object_mean_shape,
                                                      object_keypoints_mean,
                                                      camera_intrinsics));

  std::tie(init_success_flag, wTq_est) = object_feat_init->single_object_initialization(obj_obs_ptr, clones_cam);

  // ASSERT_NEAR((wTq_gt - wTq_est).norm(), 0, 6e-1)
  //   << "wTq_est: \n" << wTq_est << "\n" << "wTq_gt: \n" << wTq_gt << "\n";
  std::tie(dispR, dispt) = displacement(wTq_gt, wTq_est);
  EXPECT_NEAR(dispR, 0, 0.5)
    << "Rotation does not match. "
    << "\nwTq_est: \n" << wTq_est << "\n" << "wTq_gt: \n" << wTq_gt << "\n";
  ASSERT_NEAR(dispt, 0, 3.5e-1)
    << "translation does not match. "
    << "\nwTq_est: \n" << wTq_est << "\n" << "wTq_gt: \n" << wTq_gt << "\n";
  */
}

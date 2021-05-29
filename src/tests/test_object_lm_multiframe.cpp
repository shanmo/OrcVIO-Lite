#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
 
#include "orcvio/obj/ObjectFeature.h"
#include "orcvio/obj/ObjectFeatureInitializer.h"
#include "orcvio/obj/ObjectLM.h"
#include "orcvio/utils/se3_ops.hpp"
#include "orcvio/tests/test_utils.h"
#include <orcvio/utils/EigenNumericalDiff.h>

#ifdef HAVE_PYTHONLIBS
#include <orcvio/plot/matplotlibcpp.h>
constexpr bool DEBUG = false;
#else
constexpr bool DEBUG = false;
#endif
 
using namespace std;
using namespace Eigen;

using orcvio::dsread;
using orcvio::displacement;
using orcvio::vector_eigen;
using orcvio::CameraLM;
using orcvio::LMCameraState;

using namespace orcvio;
using orcvio::load_multi_frame_test_data;
namespace fs = boost::filesystem;
 
/**
  convert kps to bbox
  :param kps:
  :return:
*/
Vector4d kps2bbox(const MatrixX2d& kps,
                 double min_x = -1.0,
                 double min_y = -1.0,
                 double max_x = 1.0,
                 double max_y = 1.0)
{
  assert(kps.rows() >= 1);

  Vector2d min_row = kps.colwise().minCoeff();
  Vector2d max_row = kps.colwise().maxCoeff();

  // left, up, right, down
  Vector4d bbox;
  bbox << max(min_x, min_row(0)), // xmin
    max(min_y, min_row(1)), // ymin
    min(max_x, max_row(0)), // xmax
    min(max_y, max_row(1)); // ymax
  assert(bbox.allFinite());

  return bbox;
}
 
TEST(ObjectLMMultiFrame, test_object_lm_multiframe)
{
  std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
  std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;

  // initialize object feature structure
  std::shared_ptr<ObjectFeature> obj_obs_ptr;
  obj_obs_ptr.reset(new ObjectFeature(1, "car"));

  Eigen::Matrix4d wTq_gt;
  Eigen::Vector3d object_mean_shape;
  Eigen::MatrixX3d object_keypoints_mean;

  load_multi_frame_test_data("src/tests/data/one_car/frame_%d.h5",
                            clones_cam,
                            obj_obs_ptr,
                            wTq_gt,
                            object_mean_shape,
                            object_keypoints_mean);
 
  std::unique_ptr<ObjectFeatureInitializer> object_feat_init;
  Matrix3d camera_intrinsics = Matrix3d::Identity();
  FeatureInitializerOptions featinit_options;

  Vector4d residual_weights = Vector4d::Ones();
  residual_weights(1) = 3e-2;
  object_feat_init.reset(new ObjectFeatureInitializer(featinit_options,
                                                      object_mean_shape,
                                                      object_keypoints_mean,
                                                      camera_intrinsics,
                                                      residual_weights));
 
  bool init_success_flag;
  Eigen::Matrix4d wTq_est;
  std::tie(init_success_flag, wTq_est) = object_feat_init->single_object_initialization(obj_obs_ptr, clones_cam);

  ASSERT_TRUE(init_success_flag) << "object initialization failed";
  double dispR, dispt;
  std::tie(dispR, dispt) = displacement(wTq_gt, wTq_est);
  EXPECT_NEAR(dispR, 0, 0.5)
    << "Initialization is not that good. "
    << "\nwTq_est: \n" << wTq_est << "\n" << "wTq_gt: \n" << wTq_gt << "\n";
  EXPECT_NEAR(dispt, 0, 3.5e-1)
    << "Initialization is not that good. "
    << "\nwTq_est: \n" << wTq_est << "\n" << "wTq_gt: \n" << wTq_gt << "\n";
  // std::cout << "object init success, id: " << object_id << std::endl;
  ObjectState object_state;
  object_state.object_id = 0;
  object_state.object_pose = wTq_est;
  object_state.ellipsoid_shape = object_feat_init->getObjectMeanShape();
  object_state.object_keypoints_shape_global_frame = orcvio::transform_mean_keypoints_to_global(object_feat_init->getObjectKeypointsMeanShape(), wTq_est);

  bool use_left_perturbation_flag = true;
  bool use_new_bbox_residual_flag = false;
  bool success = object_feat_init->single_levenberg_marquardt(
      *obj_obs_ptr, clones_cam, object_state,
      use_left_perturbation_flag, use_new_bbox_residual_flag);
  ASSERT_TRUE(success) << "Object LM failed";
  std::tie(dispR, dispt) = displacement(wTq_gt, object_state.object_pose);
  EXPECT_NEAR(dispR, 0, 0.5 )
    << "object_pose(R):\n" << object_state.object_pose.block<3,3>(0,0) << "\n wRq_gt:\n" << wTq_gt.block<3,3>(0,0) << "\n";
  ASSERT_NEAR(dispt, 0, (0.05 * wTq_gt.topRightCorner<3,1>().norm()) )
    << "object_pose(t):\n" << object_state.object_pose.topRightCorner<3,1>() << "\n wtq_gt:\n" << wTq_gt.topRightCorner<3,1>() << "\n";

}
 
 
 
TEST(ObjectLMMultiFrame, test_error_FeatureQuadric_multiframe)
{
 
  // load single frame data
  const std::string filepath = "src/tests/data/test_error_feature_quadric.h5";
  auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
  Eigen::MatrixX2d zs_e = dsread(name2mat, "zs");
  Eigen::Matrix4d cTw = dsread(name2mat, "S");
  Eigen::Matrix4d wTo = dsread(name2mat, "T");
  auto errors_e = dsread(name2mat, "error");
  auto jacobian_e = dsread(name2mat, "jacobian");

  Eigen::Matrix<double, 12, 4> M_e = dsread(name2mat, "M");
  // dummy mean shape
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  // copy single frame data to generate multi frame data
  vector_eigen<Eigen::MatrixX2d> zs_all;
  zs_all.push_back(zs_e);
  zs_all.push_back(zs_e);

  vector_eigen<Eigen::Matrix4d> cTw_all;
  cTw_all.push_back(cTw);
  cTw_all.push_back(cTw);

  Eigen::VectorXd error_all(errors_e.cols() * 2);
  error_all << errors_e.transpose(), errors_e.transpose();

  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true;

  ObjectLM::ErrorFeatureQuadric efq(zs_all, cTw_all, camera_intrinsics,
                                    object_keypoints_mean,
                                    use_left_perturbation_flag);

  LMSE3 wTo_SE3(wTo);
  LMObjectState object(wTo_SE3, Eigen::Vector3d::Zero(), M_e);

  Eigen::VectorXd errors_est(efq.values());
  efq(object, errors_est);

  ASSERT_NEAR((error_all - errors_est).norm(), 0, 1e-6)
    << "errors_est: \n" << errors_est << "\n"
    << "errors_true: \n" << error_all << "\n"; 
 
}
 
TEST(ObjectLMMultiFrame, test_jacobian_FeatureQuadric_multiframe)
{
 
  // load single frame data
  const std::string filepath = "src/tests/data/test_error_feature_quadric.h5";
  auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
  Eigen::MatrixX2d zs_e = dsread(name2mat, "zs");
  Eigen::Matrix4d cTw = dsread(name2mat, "S");
  Eigen::Matrix4d wTo = dsread(name2mat, "T");
  auto errors_e = dsread(name2mat, "error");
  auto jacobian_e = dsread(name2mat, "jacobian");

  Eigen::Matrix<double, 12, 4> M_e = dsread(name2mat, "M");
  // dummy mean shape
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  // copy single frame data to generate multi frame data
  vector_eigen<Eigen::MatrixX2d> zs_all;
  zs_all.push_back(zs_e);
  zs_all.push_back(zs_e);

  vector_eigen<Eigen::Matrix4d> cTw_all;
  cTw_all.push_back(cTw);
  cTw_all.push_back(cTw);

  Eigen::VectorXd error_all(errors_e.cols() * 2);
  error_all << errors_e.transpose(), errors_e.transpose();

  // jacobian_e size is 24 x 45
  Eigen::MatrixXd jac_all(jacobian_e.rows() * 2, jacobian_e.cols());
  jac_all << jacobian_e, jacobian_e;

  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true;

  ObjectLM::ErrorFeatureQuadric efq(zs_all, cTw_all, camera_intrinsics,
                                    object_keypoints_mean,
                                    use_left_perturbation_flag);

  LMSE3 wTo_SE3(wTo);
  LMObjectState object(wTo_SE3, Eigen::Vector3d::Zero(), M_e);

  Eigen::MatrixXd jacobian_est(efq.values(), efq.inputs());
  efq.df(object, jacobian_est);

  ASSERT_NEAR((jac_all - jacobian_est).norm(), 0, 1e-6)
    << "jacobian_est: \n" << jacobian_est << "\n"
    << "jacobian_true: \n" << jac_all << "\n"; 
 
}
 
TEST(ObjectLMMultiFrame, test_error_BBoxQuadric_multiframe)
{
 
  // load single frame data
  const std::string filepath = "src/tests/data/test_error_bbox_quadric.h5";
  auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
  auto const zb = dsread(name2mat, "zb");
  MatrixX2d zs = dsread(name2mat, "zs");
  Eigen::Matrix4d cTw = dsread(name2mat, "S");
  Eigen::Matrix4d wTo = dsread(name2mat, "T");
  auto errors_e = dsread(name2mat, "error");
  auto jacobian_e = dsread(name2mat, "jacobian");
  auto const v = dsread(name2mat, "v");

  // dummy mean shape
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  // copy single frame data to generate multi frame data
  vector_eigen<Eigen::Vector4d> zb_all;
  zb_all.push_back(zb.transpose());
  zb_all.push_back(zb.transpose());

  vector_eigen<Eigen::MatrixX2d> zs_all;
  zs_all.push_back(zs);
  zs_all.push_back(zs);

  vector_eigen<Eigen::Matrix4d> cTw_all;
  cTw_all.push_back(cTw);
  cTw_all.push_back(cTw);

  Eigen::VectorXd error_all(errors_e.cols() * 2);
  error_all << errors_e.transpose(), errors_e.transpose();

  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true;
  bool use_new_bbox_residual_flag = false;

  ObjectLM::ErrorBBoxQuadric ebq(zs_all, zb_all, cTw_all, camera_intrinsics, object_keypoints_mean,
    use_left_perturbation_flag, use_new_bbox_residual_flag);

  LMSE3 wTo_SE3(wTo);
  LMObjectState object(wTo_SE3, v, object_keypoints_mean);

  Eigen::VectorXd errors_est(ebq.values());
  ebq(object, errors_est);

  ASSERT_NEAR((error_all - errors_est).norm(), 0, 1e-6)
    << "errors_est: \n" << errors_est << "\n"
    << "errors_true: \n" << error_all << "\n"; 
 
}
 
TEST(ObjectLMMultiFrame, test_jacobian_BBoxQuadric_multiframe)
{

  // load single frame data
  const std::string filepath = "src/tests/data/test_error_bbox_quadric.h5";
  auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
  auto const zb = dsread(name2mat, "zb");
  MatrixX2d zs = dsread(name2mat, "zs");
  Eigen::Matrix4d cTw = dsread(name2mat, "S");
  Eigen::Matrix4d wTo = dsread(name2mat, "T");
  auto errors_e = dsread(name2mat, "error");
  auto jacobian_e = dsread(name2mat, "jacobian");
  auto const v = dsread(name2mat, "v");

  // dummy mean shape
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  // copy single frame data to generate multi frame data
  vector_eigen<Eigen::Vector4d> zb_all;
  zb_all.push_back(zb.transpose());
  zb_all.push_back(zb.transpose());

  vector_eigen<Eigen::MatrixX2d> zs_all;
  zs_all.push_back(zs);
  zs_all.push_back(zs);

  vector_eigen<Eigen::Matrix4d> cTw_all;
  cTw_all.push_back(cTw);
  cTw_all.push_back(cTw);

  Eigen::VectorXd error_all(errors_e.cols() * 2);
  error_all << errors_e.transpose(), errors_e.transpose();

  Eigen::MatrixXd jac_all(jacobian_e.rows() * 2, jacobian_e.cols());
  jac_all << jacobian_e, jacobian_e;

  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true;
  bool use_new_bbox_residual_flag = false;

  ObjectLM::ErrorBBoxQuadric ebq(zs_all, zb_all, cTw_all, camera_intrinsics, object_keypoints_mean,
    use_left_perturbation_flag, use_new_bbox_residual_flag);

  LMSE3 wTo_SE3(wTo);
  LMObjectState object(wTo_SE3, v, object_keypoints_mean);

  ObjectLM::ErrorBBoxQuadric::JacobianType fjac(ebq.values(), ebq.inputs());
  ebq.df(object, fjac);

  ASSERT_NEAR((jac_all - fjac).norm(), 0, 1e-6)
    << "jacobian_est: \n" << fjac << "\n"
    << "jacobian_true: \n" << jac_all << "\n"; 
 
}
 
/*
for the sensor jacobians in multiframe case
*/
 
TEST(ObjectLMMultiFrame, test_error_sensor_FeatureQuadric_multiframe)
{
 
  // load single frame data
  const std::string filepath = "src/tests/data/test_error_feature_quadric.h5";
  auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
  Eigen::MatrixX2d zs_e = dsread(name2mat, "zs");
  Eigen::Matrix4d cTw = dsread(name2mat, "S");
  Eigen::Matrix4d wTo = dsread(name2mat, "T");
  auto errors_e = dsread(name2mat, "error");
  auto jacobian_e = dsread(name2mat, "jacobian");

  Eigen::Matrix<double, 12, 4> M_e = dsread(name2mat, "M");
  // dummy mean shape
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  // copy single frame data to generate multi frame data
  vector_eigen<Eigen::MatrixX2d> zs_all;
  zs_all.push_back(zs_e);
  zs_all.push_back(zs_e);

  vector_eigen<Eigen::Matrix4d> cTw_all;
  cTw_all.push_back(cTw);
  cTw_all.push_back(cTw);

  Eigen::VectorXd error_all(errors_e.cols() * 2);
  error_all << errors_e.transpose(), errors_e.transpose();

  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true;

  CameraLM::ErrorFeatureQuadric efq(zs_all, camera_intrinsics,
                                    object_keypoints_mean,
                                    use_left_perturbation_flag);

  LMSE3 wTo_SE3(wTo);
  LMCameraState object(wTo_SE3, Eigen::Vector3d::Zero(), 
    M_e.block<12, 4>(0, 0).eval(), cTw_all);

  Eigen::VectorXd errors_est(efq.values());
  efq(object, errors_est);

  ASSERT_TRUE(errors_est.rows() == error_all.rows())
      << "Jacobian row size(est): \n" << errors_est.rows() << "\n"
      << "Jacobian row size(true): \n" << error_all.rows() << "\n";

  ASSERT_TRUE(errors_est.cols() == error_all.cols())
      << "Jacobian col size(est): \n" << errors_est.cols() << "\n"
      << "Jacobian col size(true): \n" << error_all.cols() << "\n";

  ASSERT_NEAR((error_all - errors_est).norm(), 0, 1e-6)
    << "errors_est: \n" << errors_est << "\n"
    << "errors_true: \n" << error_all << "\n"; 
 
}

TEST(ObjectLMMultiFrame, test_sensor_jacobian_FeatureQuadric_multiframe)
{
 
  // load single frame data
  const std::string filepath = "src/tests/data/test_error_feature_quadric.h5";
  auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
  Eigen::MatrixX2d zs_e = dsread(name2mat, "zs");
  Eigen::Matrix4d cTw = dsread(name2mat, "S");
  Eigen::Matrix4d wTo = dsread(name2mat, "T");
  auto errors_e = dsread(name2mat, "error");
  auto jacobian_e = dsread(name2mat, "jacobian");

  Eigen::Matrix<double, 12, 4> M_e = dsread(name2mat, "M");
  // dummy mean shape
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  // copy single frame data to generate multi frame data
  vector_eigen<Eigen::MatrixX2d> zs_all;
  zs_all.push_back(zs_e);
  zs_all.push_back(zs_e);

  vector_eigen<Eigen::Matrix4d> cTw_all;
  cTw_all.push_back(cTw);
  cTw_all.push_back(cTw);

  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true;

  CameraLM::ErrorFeatureQuadric efq(zs_all, camera_intrinsics,
                                    object_keypoints_mean,
                                    use_left_perturbation_flag);

  LMSE3 wTo_SE3(wTo);
  LMCameraState object(wTo_SE3, Eigen::Vector3d::Zero(), object_keypoints_mean, cTw_all);

  Eigen::MatrixXd jacobian_est_test(efq.values(), efq.inputs() * cTw_all.size());
  efq.df_test(object, jacobian_est_test);

  ASSERT_TRUE(jacobian_est_test.rows() == zs_e.rows() * 4)
      << "Jacobian row size(est): \n" << jacobian_est_test.rows() << "\n"
      << "Jacobian row size(true): \n" << zs_e.rows() * 4 << "\n";

  ASSERT_TRUE(jacobian_est_test.cols() == 6 * cTw_all.size())
      << "Jacobian col size(est): \n" << jacobian_est_test.cols() << "\n"
      << "Jacobian col size(true): \n" << 6 * cTw_all.size() << "\n"; 

  EigenNumericalDiff::NumericalDiff<CameraLM::ErrorFeatureQuadric> efq_num_df_sensor(efq);
  Eigen::MatrixXd jacobian_sensor_num(efq.values(), efq.inputs() * cTw_all.size());
  efq_num_df_sensor.df(object, jacobian_sensor_num);

  ASSERT_NEAR((jacobian_sensor_num - jacobian_est_test).norm(), 0, 1e-6)
    << "jacobian_est: \n" << jacobian_est_test << "\n"
    << "jacobian_true: \n" << jacobian_sensor_num << "\n"; 

}


TEST(ObjectLMMultiFrame, test_sensor_error_BBoxQuadric_multiframe)
{

  // load single frame data
  const std::string filepath = "src/tests/data/test_error_bbox_quadric.h5";
  auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
  auto const zb = dsread(name2mat, "zb");
  MatrixX2d zs = dsread(name2mat, "zs");
  Eigen::Matrix4d cTw = dsread(name2mat, "S");
  Eigen::Matrix4d wTo = dsread(name2mat, "T");
  auto errors_e = dsread(name2mat, "error");
  auto jacobian_e = dsread(name2mat, "jacobian");
  auto const v = dsread(name2mat, "v");

  // dummy mean shape
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  // copy single frame data to generate multi frame data
  vector_eigen<Eigen::Vector4d> zb_all;
  zb_all.push_back(zb.transpose());
  zb_all.push_back(zb.transpose());

  vector_eigen<Eigen::MatrixX2d> zs_all;
  zs_all.push_back(zs);
  zs_all.push_back(zs);

  vector_eigen<Eigen::Matrix4d> cTw_all;
  cTw_all.push_back(cTw);
  cTw_all.push_back(cTw);

  Eigen::VectorXd error_all(errors_e.cols() * 2);
  error_all << errors_e.transpose(), errors_e.transpose();

  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true;
  bool use_new_bbox_residual_flag = false;

  CameraLM::ErrorBBoxQuadric ebq(zs_all, zb_all, camera_intrinsics, object_keypoints_mean,
    use_left_perturbation_flag, use_new_bbox_residual_flag);

  LMSE3 wTo_SE3(wTo);
  LMCameraState object(wTo_SE3, v, object_keypoints_mean, cTw_all);

  Eigen::VectorXd errors_est(ebq.values());
  ebq(object, errors_est);

  ASSERT_NEAR((error_all - errors_est).norm(), 0, 1e-6)
    << "errors_est: \n" << errors_est << "\n"
    << "errors_true: \n" << error_all << "\n"; 

}


TEST(ObjectLMMultiFrame, test_sensor_jacobian_BBoxQuadric_multiframe)
{
 
  // load single frame data
  const std::string filepath = "src/tests/data/test_error_bbox_quadric.h5";
  auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
  auto const zb = dsread(name2mat, "zb");
  MatrixX2d zs = dsread(name2mat, "zs");
  Eigen::Matrix4d cTw = dsread(name2mat, "S");
  Eigen::Matrix4d wTo = dsread(name2mat, "T");
  auto errors_e = dsread(name2mat, "error");
  auto const v = dsread(name2mat, "v");

  // dummy mean shape
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  // copy single frame data to generate multi frame data
  vector_eigen<Eigen::Vector4d> zb_all;
  zb_all.push_back(zb.transpose());
  zb_all.push_back(zb.transpose());

  vector_eigen<Eigen::MatrixX2d> zs_all;
  zs_all.push_back(zs);
  zs_all.push_back(zs);

  vector_eigen<Eigen::Matrix4d> cTw_all;
  cTw_all.push_back(cTw);
  cTw_all.push_back(cTw);

  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true;
  bool use_new_bbox_residual_flag = false;

  CameraLM::ErrorBBoxQuadric ebq(zs_all, zb_all, camera_intrinsics, object_keypoints_mean,
    use_left_perturbation_flag, use_new_bbox_residual_flag);

  LMSE3 wTo_SE3(wTo);
  LMCameraState object(wTo_SE3, v, object_keypoints_mean, cTw_all);

  Eigen::MatrixXd fjac(ebq.values(), ebq.inputs() * cTw_all.size());
  ebq.df_test(object, fjac);

  ASSERT_TRUE(fjac.rows() == cTw_all.size() * 4)
      << "Jacobian row size(est): \n" << fjac.rows() << "\n"
      << "Jacobian row size(true): \n" << cTw_all.size() * 4 << "\n";

  ASSERT_TRUE(fjac.cols() == 6 * cTw_all.size())
      << "Jacobian col size(est): \n" << fjac.cols() << "\n"
      << "Jacobian col size(true): \n" << 6 * cTw_all.size() << "\n";

  EigenNumericalDiff::NumericalDiff<CameraLM::ErrorBBoxQuadric> ebq_num_df_sensor(ebq);
  Eigen::MatrixXd jacobian_sensor_num(ebq.values(), ebq.inputs() * cTw_all.size());
  ebq_num_df_sensor.df(object, jacobian_sensor_num);

  ASSERT_NEAR((jacobian_sensor_num - fjac).norm(), 0, 1e-6)
    << "jacobian_est: \n" << fjac << "\n"
    << "jacobian_true: \n" << jacobian_sensor_num << "\n"; 

}
 
 
 

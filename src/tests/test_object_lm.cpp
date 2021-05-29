#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/hdf.hpp>

#include <orcvio/obj/ObjectLM.h>
#include <orcvio/obj/ObjectResJacCam.h>
#include <orcvio/utils/se3_ops.hpp>
#include <orcvio/tests/test_utils.h>
#include <orcvio/utils/EigenNumericalDiff.h>

// constexpr bool DEBUG = true;
constexpr bool DEBUG = false;

using namespace std;
//using namespace Eigen;
using Eigen::Map;
using Eigen::MatrixX3d;
using Eigen::Matrix3d;
using Eigen::MatrixX2d;
using Eigen::Matrix4d;
using Eigen::MatrixXd;
using Eigen::MatrixX4d;
using Eigen::Matrix2Xd;
using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::RowMajor;
using EigenLevenbergMarquardt::DenseFunctor;
using EigenNumericalDiff::NumericalDiff;
using orcvio::ObjectLM;
using orcvio::CameraLM;
using orcvio::LMSE3;
using orcvio::LMObjectState;
using orcvio::LMCameraState;
using orcvio::dsread;
using orcvio::vector_eigen;
using orcvio::ensure_path_exists;
namespace fs = boost::filesystem;

class ErrorFeatureQuadricTests : public ::testing::Test
{
protected:
    Eigen::MatrixXd zs_e;
    Eigen::MatrixXd cTw;
    Eigen::MatrixXd errors_e;
    Eigen::MatrixXd jacobian_e;
    std::unique_ptr<const ObjectLM::ErrorFeatureQuadric> erq_ptr;
    LMObjectState object;
    virtual void SetUp() {
      
       const std::string filepath = "src/tests/data/test_error_feature_quadric.h5";
       auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
       zs_e = dsread(name2mat, "zs");
       cTw = dsread(name2mat, "S");
       auto const wTo = dsread(name2mat, "T");
       errors_e = dsread(name2mat, "error");
       jacobian_e = dsread(name2mat, "jacobian");
       auto const M_e = dsread(name2mat, "M");

       // dummy mean shape 
       Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

       LMSE3 wTo_SE3(wTo);
       LMObjectState _object(wTo_SE3,
                            Eigen::Vector3d::Zero(),
                            M_e.block<12, 4>(0, 0).eval());
       object = _object;

       Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();
       if (DEBUG) {
           std::cerr << "good inputs: " << cTw.topRows<3>() << wTo << M_e << "\n";
           Eigen::MatrixX2d uvs = orcvio::project_object_points(cTw.topRows<3>(), wTo, M_e);
           std::cerr << "good uvs: " << uvs << "\n";
           auto zs_predict = uvs - Map<const Matrix<double, Dynamic, 2, RowMajor>>(errors_e.data(), zs_e.rows(), zs_e.cols());
           std::cerr << "zs: \n" << zs_e << "\n"
                     << "zs_predict: \n" << zs_predict << "\n";
       }

       bool use_left_perturbation_flag = true;
       //  bool use_left_perturbation_flag = false;

       auto erqptr = new ObjectLM::ErrorFeatureQuadric({zs_e}, {cTw}, camera_intrinsics,
                                         object_keypoints_mean,
                                         use_left_perturbation_flag);
       erq_ptr.reset(erqptr);
    }
};

TEST_F( ErrorFeatureQuadricTests, testErrorFeatureQuadric) {

  // Eigen::VectorXd errors_est_obj =  erq(object, 0);
  // ASSERT_NEAR((errors_e.transpose() - errors_est_obj).norm(), 0, 1e-6)
  //   << "errors_est_obj: \n" << errors_est_obj << "\n"
  //   << "errors_true: \n" << errors_e.transpose() << "\n";

  auto const& erq = *erq_ptr;
  Eigen::VectorXd errors_est(erq.values());
  erq(object, errors_est);
  ASSERT_NEAR((errors_e.transpose() - errors_est).norm(), 0, 1e-6)
    << "errors_est: \n" << errors_est << "\n"
    << "errors_true: \n" << errors_e.transpose() << "\n";

  Eigen::MatrixXd jacobian(erq.values(), erq.inputs());
  erq.df(object, jacobian);
  ASSERT_NEAR( (jacobian.block(0, 0, 2, 6)
                - jacobian_e.block(0, 0, 2, 6)
                ).norm(),
               0, 1e-6)
    << "jacobian_est: " << jacobian.block(0, 0, 3, 7) << "\n"
    << "jacobian_true: " << jacobian_e.block(0, 0, 3, 7) << "\n";

  for (int i = 0; i < zs_e.rows(); ++i) {
    ASSERT_NEAR( (jacobian.block(2*i, 0, 2, 6)
                  - jacobian_e.block(2*i, 0, 2, 6)
                  ).norm(),
                0, 1e-6)
      << "jacobian_est(" << i << "): " << jacobian.block(2*i, 0, 3, 7) << "\n"
      << "jacobian_true(" << i<< "): " << jacobian_e.block(2*i, 0, 3, 7) << "\n";
  }

  ASSERT_NEAR( (jacobian.block(0, 6, 2, 6)
                - jacobian_e.block(0, 6, 2, 6)
                ).norm(),
               0, 1e-6)
    << "jacobian_est: " << jacobian.block(0, 6, 3, 7) << "\n"
    << "jacobian_true: " << jacobian_e.block(0, 6, 3, 7) << "\n";

  ASSERT_NEAR( (jacobian.block(0, 9, 2, 3)
                - jacobian_e.block(0, 9, 2, 3)
                ).norm(),
               0, 1e-6)
    << "jacobian_est: " << jacobian.block(0, 9, 3, 4) << "\n"
    << "jacobian_true: " << jacobian_e.block(0, 9, 3, 4) << "\n";

  for (int i = 0; i < zs_e.rows(); ++i) {
    ASSERT_NEAR( (jacobian.block(2*i, 3*i+9, 2, 3)
                  - jacobian_e.block(2*i, 3*i+9, 2, 3)
                  ).norm(),
                0, 1e-6)
      << "jacobian_est( " << 2*i << ", " << 3*i+9 << "): " << jacobian.block(2*i, 3*i+9, 3, 4) << "\n"
      << "jacobian_true( " << 2*i << ", " << 3*i+9 << "): " << jacobian_e.block(2*i, 3*i+9, 3, 4) << "\n";
  }

  ASSERT_NEAR( (jacobian - jacobian_e
                ).norm(),
               0, 1e-6)
    << "jacobian_est: " << jacobian << "\n"
    << "jacobian_true: " << jacobian_e << "\n";


}

TEST( ObjectLM, testErrorBBoxQuadric) {

  auto name2mat = cv::hdf::open(ensure_path_exists("src/tests/data/test_error_bbox_quadric.h5"));
  auto const zb = dsread(name2mat, "zb");
  auto const zs = dsread(name2mat, "zs");
  auto const cTw = dsread(name2mat, "S");
  auto const wTo = dsread(name2mat, "T");
  auto const v = dsread(name2mat, "v");
  auto const error = dsread(name2mat, "error");
  auto const jacobian = dsread(name2mat, "jacobian");
  
  // dummy mean shape 
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  LMSE3 wTo_SE3(wTo);
  LMObjectState object(wTo_SE3 ,
                                v,
                                Eigen::Matrix<double, 12, 3>::Zero().eval());
  
  assert(object.allFinite());
  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true; 
  bool use_new_bbox_residual_flag = false; 

  ObjectLM::ErrorBBoxQuadric ebq({zs}, {zb.transpose()}, {cTw}, camera_intrinsics, object_keypoints_mean,
    use_left_perturbation_flag, use_new_bbox_residual_flag);
  // Eigen::VectorXd errors_obj = ebq(object, 0);
  // ASSERT_NEAR((error.transpose() - errors_obj).norm(), 0, 1e-6)
  //   << "errors_obj: \n" << errors_obj << "\n"
  //   << "errors_true: \n" << error.transpose() << "\n";

  assert(object.allFinite());
  Eigen::VectorXd errors_est(ebq.values());
  ebq(object, errors_est);
  ASSERT_NEAR((error.transpose() - errors_est).norm(), 0, 1e-6)
    << "errors_est: \n" << errors_est << "\n"
    << "errors_true: \n" << error.transpose() << "\n";

  ObjectLM::ErrorBBoxQuadric::JacobianType fjac(ebq.values(), ebq.inputs());
  ebq.df(object, fjac);
  ASSERT_NEAR((fjac.block(0, 0, 1, 6) - jacobian.block(0, 0, 1, 6)).norm(), 0, 1e-6)
    << "fjac(0:1, 0:6): \n" << fjac.block(0, 0, 1, 6)  << "\n"
    << "jacobian(0:1, 0:6): \n" << jacobian.block(0, 0, 1, 6)  << "\n";

  ASSERT_NEAR((fjac - jacobian).norm(), 0, 1e-6)
    << "fjac: \n" << fjac  << "\n"
    << "jacobian: \n" << jacobian  << "\n";
}

TEST(ObjectLM, test_bbox2poly) {
  Eigen::Vector4d bbox{0, 0, 20, 10};
  Eigen::MatrixX2d poly(4, 2);
  poly << 0, 0,
      20, 0,
      20, 10,
      0, 10;
  ASSERT_EQ(orcvio::bbox2poly(bbox), poly);
}


TEST(ObjectLM, test_poly2lines) {
  Eigen::MatrixX2d poly(4, 2);
  poly << 0, 0,
    0, 10,
    20, 10,
    20, 0;
  Eigen::MatrixX3d linesh = orcvio::poly2lineh(poly);
  Eigen::MatrixX3d poly_hom = Eigen::MatrixX3d::Ones(4, 3);
  poly_hom.block(0, 0, poly.rows(), poly.cols()) = poly;
  for (int i = 0; i < poly.rows(); ++i) {
    double doti = linesh.row(i).dot(poly_hom.row(i));
    ASSERT_EQ(doti, 0.) << "dot" << i << ": " << doti;
    int ip1 = (i + 1) % poly.rows();
    double dotip1 = linesh.row(ip1).dot(poly_hom.row(ip1));
    ASSERT_EQ(dotip1, 0.) << "dot" << i << "p1: " << dotip1;
  }
}

TEST( ObjectLM, testErrorDeformReg) {
  
  auto name2mat = cv::hdf::open(ensure_path_exists("src/tests/data/test_error_deform_reg.h5"));
  auto const zs = dsread(name2mat, "zs");
  auto const M = dsread(name2mat, "M");
  auto const Mhat = dsread(name2mat, "Mhat");
  auto const error = dsread(name2mat, "error");
  auto const jacobian = dsread(name2mat, "jacobian");

  LMSE3 wTo_SE3;
  LMObjectState object(wTo_SE3 ,
                      Eigen::Vector3d::Zero(),
                      M.block<12, 4>(0,0).eval());

  ObjectLM::ErrorDeformRegularization edr({zs}, Mhat);

  Eigen::VectorXd fvec(edr.values());
  edr(object, fvec);
  ASSERT_NEAR((fvec - error).norm(), 0, 1e-6)
    << "error(got): \n" << fvec << "\n"
    << "error(true): \n" << error << "\n";

  ObjectLM::ErrorDeformRegularization::JacobianType fjac(edr.values(), edr.inputs());
  edr.df(object, fjac);
  
  ASSERT_NEAR((fjac - jacobian).norm(), 0, 1e-6)
    << "jacobian(got): \n" << fjac << "\n"
    << "jacobian(true): \n" << jacobian << "\n";

}

TEST( ObjectLM, testErrorQuadVReg) {
  auto name2mat = cv::hdf::open(ensure_path_exists("src/tests/data/test_error_mean_shape_reg.h5"));
  auto const zs = dsread(name2mat, "zs");
  auto const v = dsread(name2mat, "v");
  auto const mean_v = dsread(name2mat, "mean_v");
  auto const error = dsread(name2mat, "error");
  auto const jacobian = dsread(name2mat, "jacobian");

  // dummy mean shape 
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  LMSE3 wTo_SE3;
  LMObjectState object(wTo_SE3 ,
                      v,
                      object_keypoints_mean);

  ObjectLM::ErrorQuadVRegularization eqvr({zs}, object_keypoints_mean, mean_v);
  Eigen::VectorXd fvec(eqvr.values());
  eqvr(object, fvec);
  ASSERT_NEAR((fvec - error).norm(), 0, 1e-6)
    << "error(got): \n" << fvec << "\n"
    << "error(true): \n" << error << "\n";

  ObjectLM::ErrorDeformRegularization::JacobianType fjac(eqvr.values(), eqvr.inputs());
  eqvr.df(object, fjac);
  ASSERT_NEAR((fjac - jacobian).norm(), 0, 1e-6)
    << "jacobian(got): \n" << fjac << "\n"
    << "jacobian(true): \n" << jacobian << "\n";

}

TEST_F(ErrorFeatureQuadricTests, testErrorFeatureJacobianNumerical)
{

  auto const& erq = *erq_ptr;

  Eigen::MatrixXd jacobian(erq.values(), erq.inputs());
  erq.df(object, jacobian);

  Eigen::MatrixXd jacobian_num(erq.values(), erq.inputs());
  NumericalDiff<ObjectLM::ErrorFeatureQuadric> erq_num(erq);
  erq_num.df(object, jacobian_num);

  double atol = 1e-6;
  double rtol = 1e-4;
  auto bool_mat = ((jacobian - jacobian_num).array().abs()
                   <= (atol + rtol * jacobian_num.array().abs()));
  int pnr = 6; // print n rows at a time
  int pnc = 3; // print n cols at a time
  for (int r = 0; r < jacobian.rows() / pnr; r++) {
      for (int c = 0; c < jacobian.cols() / pnc; c++) {
          EXPECT_TRUE(bool_mat.block(r*pnr, c*pnc, pnr, pnc).all())
              << "evaluated.block(" << r*pnr << "," << c*pnc << ", pnr, pnc) : \n" << jacobian.block(r*pnr, c*pnc, pnr, pnc) << "\n"
              << "numerical.block(" << r*pnr << "," << c*pnc << ", pnr, pnc) : \n" << jacobian_num.block(r*pnr, c*pnc, pnr, pnc) << "\n";
      }
  }
  ASSERT_TRUE(bool_mat.all())
      << "evaluated : \n" << jacobian << "\n"
      << "numerical : \n" << jacobian_num << "\n";
}

TEST( ObjectLM, testErrorBBoxJacobianNumerical) {

  auto name2mat = cv::hdf::open(ensure_path_exists("src/tests/data/test_error_bbox_quadric.h5"));
  auto const zb = dsread(name2mat, "zb");
  auto const zs = dsread(name2mat, "zs");
  auto const cTw = dsread(name2mat, "S");
  auto const wTo = dsread(name2mat, "T");
  auto const v = dsread(name2mat, "v");

  // dummy mean shape 
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  LMSE3 wTo_SE3(wTo);
  LMObjectState object(wTo_SE3 ,
                                v,
                                Eigen::Matrix<double, 12, 3>::Zero().eval());
  
  assert(object.allFinite());
  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true; 
  bool use_new_bbox_residual_flag = false; 

  ObjectLM::ErrorBBoxQuadric ebq({zs}, {zb.transpose()}, {cTw}, camera_intrinsics, object_keypoints_mean,
    use_left_perturbation_flag, use_new_bbox_residual_flag);

  Eigen::MatrixXd jacobian(ebq.values(), ebq.inputs());
  ebq.df(object, jacobian);

  Eigen::MatrixXd jacobian_num(ebq.values(), ebq.inputs());
  NumericalDiff<ObjectLM::ErrorBBoxQuadric> ebq_num(ebq);
  ebq_num.df(object, jacobian_num);

  double atol = 1e-6;
  double rtol = 1e-4;
  auto bool_mat = ((jacobian - jacobian_num).array().abs()
                   <= (atol + rtol * jacobian_num.array().abs()));
  ASSERT_TRUE(bool_mat.all())
      << "evaluated : \n" << jacobian << "\n"
      << "numerical : \n" << jacobian_num << "\n";
}


TEST( ObjectLM, testErrorDeformRegJacobianNumerical) {
  
  auto name2mat = cv::hdf::open(ensure_path_exists("src/tests/data/test_error_deform_reg.h5"));
  auto const zs = dsread(name2mat, "zs");
  auto const M = dsread(name2mat, "M");
  auto const Mhat = dsread(name2mat, "Mhat");

  LMSE3 wTo_SE3;
  LMObjectState object(wTo_SE3 ,
                      Eigen::Vector3d::Zero(),
                      M.block<12, 4>(0,0).eval());

  ObjectLM::ErrorDeformRegularization edr({zs}, Mhat);

  ObjectLM::ErrorDeformRegularization::JacobianType jacobian(edr.values(), edr.inputs());
  edr.df(object, jacobian);

  Eigen::MatrixXd jacobian_num(edr.values(), edr.inputs());
  NumericalDiff<ObjectLM::ErrorDeformRegularization> edr_num(edr);
  edr_num.df(object, jacobian_num);

  double atol = 1e-6;
  double rtol = 1e-4;
  auto bool_mat = ((jacobian - jacobian_num).array().abs()
                   <= (atol + rtol * jacobian_num.array().abs()));
  ASSERT_TRUE(bool_mat.all())
      << "evaluated : \n" << jacobian << "\n"
      << "numerical : \n" << jacobian_num << "\n";
}

TEST( ObjectLM, testErrorQuadVRegJacobianNumerical) {
    auto name2mat = cv::hdf::open(ensure_path_exists("src/tests/data/test_error_mean_shape_reg.h5"));
    auto const zs = dsread(name2mat, "zs");
    auto const v = dsread(name2mat, "v");
    auto const mean_v = dsread(name2mat, "mean_v");

    // dummy mean shape 
    Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

    LMSE3 wTo_SE3;
    LMObjectState object(wTo_SE3 ,
                         v,
                         object_keypoints_mean);

    ObjectLM::ErrorQuadVRegularization eqvr({zs}, object_keypoints_mean, mean_v);
    ObjectLM::ErrorDeformRegularization::JacobianType jacobian(eqvr.values(), eqvr.inputs());
    eqvr.df(object, jacobian);

    Eigen::MatrixXd jacobian_num(eqvr.values(), eqvr.inputs());
    NumericalDiff<ObjectLM::ErrorQuadVRegularization> eqvr_num(eqvr);
    eqvr_num.df(object, jacobian_num);

    double atol = 1e-6;
    double rtol = 1e-4;
    auto bool_mat = ((jacobian - jacobian_num).array().abs()
                     <= (atol + rtol * jacobian_num.array().abs()));
    ASSERT_TRUE(bool_mat.all())
        << "evaluated : \n" << jacobian << "\n"
        << "numerical : \n" << jacobian_num << "\n";
}

class SensorErrorFeatureQuadricTests : public ::testing::Test
{
protected:
    Eigen::MatrixXd zs_e;
    Eigen::MatrixXd cTw;
    Eigen::MatrixXd errors_e;
    Eigen::MatrixXd jacobian_e;

    std::unique_ptr<const CameraLM::ErrorFeatureQuadric> erq_ptr;
    LMCameraState sensor_object;
    virtual void SetUp() {
      
       const std::string filepath = "src/tests/data/test_error_feature_quadric.h5";
       auto name2mat = cv::hdf::open(ensure_path_exists(filepath));
       zs_e = dsread(name2mat, "zs");
       cTw = dsread(name2mat, "S");
       auto const wTo = dsread(name2mat, "T");
       errors_e = dsread(name2mat, "error");
       jacobian_e = dsread(name2mat, "jacobian");
       auto const M_e = dsread(name2mat, "M");

       // dummy mean shape 
       Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

       LMSE3 wTo_SE3(wTo);
       LMCameraState _object(wTo_SE3,
                            Eigen::Vector3d::Zero(),
                             M_e.block<12, 4>(0, 0).eval(),
                             {cTw});
       sensor_object = _object;

       Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

       if (DEBUG)
       {
        // uvs and zs_gt should be equal 
        Eigen::MatrixX2d uvs = orcvio::project_object_points(cTw.topRows<3>(), wTo, M_e);
        std::cerr << "uvs: " << uvs << "\n";
        auto zs_gt = zs_e + Map<const Matrix<double, Dynamic, 2, RowMajor>>(errors_e.data(), zs_e.rows(), zs_e.cols());
        std::cerr << "zs_gt: \n" << zs_gt << "\n";
       }

       bool use_left_perturbation_flag = true;
       //  bool use_left_perturbation_flag = false;

       auto erqptr = new CameraLM::ErrorFeatureQuadric({zs_e}, camera_intrinsics,
                                         object_keypoints_mean,
                                         use_left_perturbation_flag);

       erq_ptr.reset(erqptr);
    }
};

TEST_F(SensorErrorFeatureQuadricTests, testErrorFeatureSensor) {

  auto const& erq = *erq_ptr;
  Eigen::VectorXd errors_est(erq.values());
  erq(sensor_object, errors_est);
  ASSERT_NEAR((errors_e.transpose() - errors_est).norm(), 0, 1e-6)
    << "errors_est: \n" << errors_est << "\n"
    << "errors_true: \n" << errors_e.transpose() << "\n";

}

TEST_F(SensorErrorFeatureQuadricTests, testErrorFeatureSensorJacobian) {

  auto const& erq = *erq_ptr;

  // Check Jacobian wrt to IMU
  Eigen::MatrixXd jacobian_sensor(erq.values(), erq.inputs());
  erq.df_test(sensor_object, jacobian_sensor);

  NumericalDiff<CameraLM::ErrorFeatureQuadric> erq_num_df_sensor(erq);
  Eigen::MatrixXd jacobian_sensor_num(erq.values(), erq.inputs());
  erq_num_df_sensor.df(sensor_object, jacobian_sensor_num);
  double atol = 1e-6;
  double rtol = 1e-4;
  auto bool_sensor_mat = ((jacobian_sensor - jacobian_sensor_num).array().abs()
                          <= (atol + rtol * jacobian_sensor_num.array().abs()));

  // std::cout  << "Jacobian row size: " << jacobian_sensor.rows() << "\n";
  // std::cout  << "Jacobian col size: " << jacobian_sensor.cols() << "\n";

  ASSERT_TRUE(jacobian_sensor.rows() == jacobian_sensor_num.rows())
      << "Jacobian row size(evaluated): \n" << jacobian_sensor.rows() << "\n"
      << "Jacobian row size(numerical): \n" << jacobian_sensor_num.rows() << "\n";

  ASSERT_TRUE(jacobian_sensor.cols() == jacobian_sensor_num.cols())
      << "Jacobian col size(evaluated): \n" << jacobian_sensor.cols() << "\n"
      << "Jacobian col size(numerical): \n" << jacobian_sensor_num.cols() << "\n";

  int pnc = 6; // print n cols at a time
  for (int r = 0; r < erq.numFrames() ; ++r) {
      int pnr = erq.block_start_frame(r+1) - erq.block_start_frame(r);

      // std::cout  << "r: " << r << "\n";
      // std::cout  << "block_start_frame(r): " << erq.block_start_frame(r) << "\n";
      // std::cout  << "block_start_frame(r+1): " << erq.block_start_frame(r+1) << "\n";

      for (int c = 0; c < bool_sensor_mat.cols() / pnc; ++c) {
          ASSERT_TRUE(bool_sensor_mat.block(r*pnr, c*pnc, pnr, pnc).all())
              << "evaluated.block(" << r*pnr << "," << c*pnc
              << ", " << pnr << ", " << pnc << ") : \n" <<
              jacobian_sensor.block(r*pnr, c*pnc, pnr, pnc) << "\n"
              << "numerical.block(" << r*pnr << "," << c*pnc
              << ", " << pnr << ", " << pnc << ") : \n" <<
              jacobian_sensor_num.block(r*pnr, c*pnc, pnr, pnc) << "\n";
      }
  }

  ASSERT_TRUE(bool_sensor_mat.all());
  // ASSERT_TRUE(bool_sensor_mat.all())
  //     << "count mismatch: " << bool_sensor_mat.count() << "\n"
  //     << "evaluated : \n" << jacobian_sensor << "\n"
  //     << "numerical : \n" << jacobian_sensor_num << "\n";

}


TEST( ObjectLM, testErrorBBoxSensor) {

  auto name2mat = cv::hdf::open(ensure_path_exists("src/tests/data/test_error_bbox_quadric.h5"));
  auto const zb = dsread(name2mat, "zb");
  auto const zs = dsread(name2mat, "zs");
  auto const cTw = dsread(name2mat, "S");
  auto const wTo = dsread(name2mat, "T");
  auto const v = dsread(name2mat, "v");
  auto const error = dsread(name2mat, "error");
  auto const jacobian = dsread(name2mat, "jacobian");
  
  // dummy mean shape 
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  LMSE3 wTo_SE3(wTo);
  LMCameraState object(wTo_SE3,
                      v,
                      Eigen::Matrix<double, 12, 3>::Zero().eval(),
                      {cTw});

  assert(object.allFinite());
  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true; 
  bool use_new_bbox_residual_flag = false; 

  CameraLM::ErrorBBoxQuadric ebq({zs}, {zb.transpose()}, camera_intrinsics, object_keypoints_mean,
    use_left_perturbation_flag, use_new_bbox_residual_flag);

  assert(object.allFinite());
  Eigen::VectorXd errors_est(ebq.values());
  ebq(object, errors_est);
  ASSERT_NEAR((error.transpose() - errors_est).norm(), 0, 1e-6)
    << "errors_est: \n" << errors_est << "\n"
    << "errors_true: \n" << error.transpose() << "\n";

}


TEST( ObjectLM, testErrorBBoxSensorJacobian) {

  auto name2mat = cv::hdf::open(ensure_path_exists("src/tests/data/test_error_bbox_quadric.h5"));
  auto const zb = dsread(name2mat, "zb");
  auto const zs = dsread(name2mat, "zs");
  auto const cTw = dsread(name2mat, "S");
  auto const wTo = dsread(name2mat, "T");
  auto const v = dsread(name2mat, "v");

  // dummy mean shape 
  Eigen::Matrix<double, 12, 3> object_keypoints_mean = Eigen::MatrixX3d::Zero(12, 3);

  LMSE3 wTo_SE3(wTo);
  LMCameraState object(wTo_SE3,
                      v,
                      Eigen::Matrix<double, 12, 3>::Zero().eval(),
                      {cTw});

  assert(object.allFinite());
  Eigen::Matrix3d camera_intrinsics = Eigen::Matrix3d::Identity();

  bool use_left_perturbation_flag = true; 
  bool use_new_bbox_residual_flag = false; 

  CameraLM::ErrorBBoxQuadric ebq({zs}, {zb.transpose()}, camera_intrinsics, object_keypoints_mean,
    use_left_perturbation_flag, use_new_bbox_residual_flag);

  Eigen::MatrixXd jacobian(ebq.values(), ebq.inputs());
  ebq.df(object, jacobian);

  Eigen::MatrixXd jacobian_num(ebq.values(), ebq.inputs());
  NumericalDiff<CameraLM::ErrorBBoxQuadric> ebq_num(ebq);
  ebq_num.df(object, jacobian_num);

  double atol = 1e-6;
  double rtol = 1e-4;
  auto bool_mat = ((jacobian - jacobian_num).array().abs()
                   <= (atol + rtol * jacobian_num.array().abs()));
  ASSERT_TRUE(bool_mat.all())
      << "evaluated : \n" << jacobian << "\n"
      << "numerical : \n" << jacobian_num << "\n";
}

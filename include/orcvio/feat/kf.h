#ifndef KF_H_
#define KF_H_

#include "Eigen/Dense"

#include <orcvio/utils/se3_ops.hpp>

using namespace orcvio; 

// we only measure the 2d position 
class MeasurementPackage {
public:
  double timestamp_;
  Eigen::VectorXd raw_measurements_;
};

class KalmanFilter {

private:
  /**
  *   Common calculation for KF and EKF.
  *   @param y = residual.
  */
  void UpdateWithResidual(const Eigen::VectorXd &y);

  // check whether the tracking toolbox was initialized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  double previous_timestamp_;

public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
  * Constructor.
  */
  KalmanFilter();

  /**
  * Destructor.
  */
  ~KalmanFilter();

  /**
  * Run the whole flow of the Kalman Filter from here.
  */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  // state vector is x, y, vx, vy 
  Eigen::VectorXd x_;

  // state covariance matrix
  Eigen::MatrixXd P_;

  // state transition matrix
  Eigen::MatrixXd F_;

  // process covariance matrix
  Eigen::MatrixXd Q_;

  // measurement matrix
  Eigen::MatrixXd H_;

  // measurement covariance matrix
  Eigen::MatrixXd R_;

  /**
   * Prediction Predicts the state and the state covariance
   * using the process model
   * @param delta_T Time between k and k+1 in s
   */
  void Predict();

  /**
   * Updates the state by using standard Kalman Filter equations
   * @param z The measurement at k+1
   */
  void Update(const Eigen::VectorXd &z);

  // to keep all tracked kps 
  vector_eigen<Eigen::Vector2f> kp_history;

};

#endif /* KF_H_ */
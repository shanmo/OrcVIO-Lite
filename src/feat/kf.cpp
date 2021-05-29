#include "orcvio/feat/kf.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {

  is_initialized_ = false;
  previous_timestamp_ = 0;

  // initializing matrices
  x_ = VectorXd(4);
  R_ = MatrixXd(2, 2);
  H_ = MatrixXd(2, 4);
  H_ << 1, 0, 0, 0,
        0, 1, 0, 0;

  //measurement covariance matrix 
  R_ << 0.0225, 0,
        0, 0.0225;

  // Initializing P
  P_ = MatrixXd(4, 4);
  P_ << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1000, 0,
        0, 0, 0, 1000;

  F_ = MatrixXd(4, 4);
  Q_ = MatrixXd(4, 4);

}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Predict() {
  x_ = F_ * x_ ;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
  */
  VectorXd y = z - H_ * x_;
  UpdateWithResidual(y);
}

void KalmanFilter::UpdateWithResidual(const VectorXd &y){
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K =  P_ * Ht * Si;
  // New state
  x_ = x_ + (K * y);
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::ProcessMeasurement(const MeasurementPackage &meas_package) {

    /*****************************************************************************
    *  Initialization
    ****************************************************************************/
    if (!is_initialized_) {
      /**
      Initialize state.
      */
      // No velocity and coordinates are cartesian already.
      const double v_init = 3 / 0.1; 
      x_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1], v_init, v_init;

      // Saving first timestamp in seconds
      previous_timestamp_ = meas_package.timestamp_ ;
      // done initializing, no need to predict or update
      is_initialized_ = true;

      return;
    }

    /**
      * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
      * Update the process noise covariance matrix.
      * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
    */
    double dt = (meas_package.timestamp_ - previous_timestamp_) / 1;
    previous_timestamp_ = meas_package.timestamp_;

    // State transition matrix update
    F_ << 1, 0, dt, 0,
      0, 1, 0, dt,
      0, 0, 1, 0,
      0, 0, 0, 1;

    // Noise covariance matrix computation
    // Noise values from the task
    double noise_ax = 9.0;
    double noise_ay = 9.0;

    double dt_2 = dt * dt; //dt^2
    double dt_3 = dt_2 * dt; //dt^3
    double dt_4 = dt_3 * dt; //dt^4
    double dt_4_4 = dt_4 / 4; //dt^4/4
    double dt_3_2 = dt_3 / 2; //dt^3/2

    Q_ << dt_4_4 * noise_ax, 0, dt_3_2 * noise_ax, 0,
          0, dt_4_4 * noise_ay, 0, dt_3_2 * noise_ay,
          dt_3_2 * noise_ax, 0, dt_2 * noise_ax, 0,
          0, dt_3_2 * noise_ay, 0, dt_2 * noise_ay;

    /*****************************************************************************
    *  Prediction
    ****************************************************************************/

    Predict();

    /*****************************************************************************
    *  Update
    ****************************************************************************/

    /**
      * Use the sensor type to perform the update step.
      * Update the state and covariance matrices.
    */

    if (meas_package.raw_measurements_[0] == 0 && meas_package.raw_measurements_[1] == 0) {
      // skip update when this keypoint is not detected 
      // note that 0, 0 is the dummy value in this case 
    } else {
      Update(meas_package.raw_measurements_);
    }

}



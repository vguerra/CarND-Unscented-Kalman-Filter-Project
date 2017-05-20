#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

//https://discussions.udacity.com/t/numerical-instability-of-the-implementation/230449/7?u=victor_guerra_986699

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.6;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // State vector's dimension
  n_x_ = 5;

  //
  n_aug_ = 7;
  n_z_ = 3;

  // Number of augmented sigma points
  n_sigma_ = 2 * n_aug_ + 1;

  // Sigma points spreading parameter
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = MatrixXd(n_x_, n_sigma_);

  // State covariance Matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  //
  R_radar_ = MatrixXd::Zero(n_z_, n_z_);
  R_radar_(0, 0) = std_radr_*std_radr_;
  R_radar_(1, 1) = std_radphi_*std_radphi_;
  R_radar_(2, 2) = std_radrd_*std_radrd_;

  R_laser_ = MatrixXd::Zero(2, 2);
  R_laser_(0, 0) = std_laspx_*std_laspx_;
  R_laser_(1, 1) = std_laspy_*std_laspy_;

  z_pred_ = VectorXd(n_z_);
  S_ = MatrixXd(n_z_, n_z_);
  Zsig_ = MatrixXd(n_z_, n_sigma_);

  H_laser_ = MatrixXd(2, n_x_);
  H_laser_ << 1, 0, 0, 0, 0,
    0, 1, 0, 0, 0;

  I_ = MatrixXd::Identity(n_x_, n_x_);

  weights_ = VectorXd(n_sigma_);
  weights_(0) = lambda_/(n_aug_ + lambda_);
  for (int i = 1; i < n_sigma_; ++i) {
    weights_(i) = 0.5/(n_aug_ + lambda_);
  }

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    time_us_ = meas_package.timestamp_;
    x_.fill(0.0);

    if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      float ro = meas_package.raw_measurements_(0);
      float phi = meas_package.raw_measurements_(1);

      float px = ro*cos(phi);
      float py = ro*sin(phi);

      x_(0) = px;
      x_(1) = py;
    }

    is_initialized_ = true;
    return;
  }

  double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0;

  if (fabs(delta_t) > 0.001) {
      Prediction(delta_t);
  }

  if (use_laser_ && meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  } else if (use_radar_ && meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);
  }

  time_us_ = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  MatrixXd Xsig_aug = AugmentedSigmaPoints();
  SigmaPointPrediction(Xsig_aug, delta_t);
  PredictMeanAndCovariance();
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  VectorXd x_pred = H_laser_ * x_;
  VectorXd diff = meas_package.raw_measurements_ - x_pred;

  MatrixXd Ht = H_laser_.transpose();
  MatrixXd PHt = P_ * Ht;
  MatrixXd S = (H_laser_ * PHt) + R_laser_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  x_ = x_ + (K * diff);
  P_ = (I_ - K * H_laser_) * P_;

  NIS_laser_ = diff.transpose() * Si * diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  PredictRadarMeasurement();
  UpdateState(meas_package.raw_measurements_);

  VectorXd diff = meas_package.raw_measurements_ - z_pred_;
  NIS_radar_ = diff.transpose() * S_.inverse() * diff;
}

MatrixXd UKF::AugmentedSigmaPoints() {
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sigma_);

  x_aug.head(n_x_) = x_;

  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_aug_ - 2, n_aug_ - 2) = std_a_ * std_a_;
  P_aug(n_aug_ - 1, n_aug_ - 1) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  MatrixXd commonTerm = sqrt(lambda_ + n_aug_) * A;

  for (int i = 1; i <= n_aug_; i++) {
    Xsig_aug.col(i) = x_aug + commonTerm.col(i - 1);
    Xsig_aug.col(n_aug_ + i) = x_aug - commonTerm.col(i - 1);
  }

  return Xsig_aug;
}

void UKF::SigmaPointPrediction(const MatrixXd& Xsig_aug, double delta_t) {

  double delta_t_2 = delta_t * delta_t;

  for (int i = 0; i < n_sigma_; ++i) {

    float v_k = Xsig_aug(2, i);
    float yawn = Xsig_aug(3, i);
    float yawn_dot = Xsig_aug(4, i);
    float mu_a_k = Xsig_aug(5, i);
    float mu_y_k = Xsig_aug(6, i);

    float sin_yawn = sin(yawn);
    float cos_yawn = cos(yawn);

    VectorXd noise(n_x_);

    noise << 0.5*delta_t_2*cos_yawn*mu_a_k,
      0.5*delta_t_2*sin_yawn*mu_a_k,
      delta_t*mu_a_k,
      0.5*delta_t_2*mu_y_k,
      delta_t*mu_y_k;

    VectorXd middle(n_x_);

    if (fabs(yawn_dot) > 0.001) {
      float angle = yawn + yawn_dot*delta_t;
      middle << (v_k/yawn_dot)*(sin(angle) - sin_yawn),
      (v_k/yawn_dot)*(-cos(angle) + cos_yawn),
      0,
      yawn_dot*delta_t,
      0;
    } else {
      middle << v_k*cos_yawn*delta_t,
      v_k*sin_yawn*delta_t,
      0,
      0,
      0;
    }

    Xsig_pred_.col(i) = Xsig_aug.col(i).head(n_x_) + middle + noise;
  }
}

void UKF::PredictMeanAndCovariance() {
  VectorXd x = VectorXd::Zero(n_x_);
  MatrixXd P = MatrixXd::Zero(n_x_, n_x_);

  for (int i = 0; i < n_sigma_; i++) {
    x += weights_(i) * Xsig_pred_.col(i);
  }

  for (int i = 0; i < n_sigma_; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    x_diff(3) = Tools::NormalizeAngle(x_diff(3));
    P += weights_(i) * x_diff * x_diff.transpose() ;
  }

  x_ = x;
  P_ = P;
}

void UKF::PredictRadarMeasurement() {
  //transform sigma points into measurement space
  for (int j = 0; j < n_sigma_; ++j) {
    float px = Xsig_pred_(0, j);
    float py = Xsig_pred_(1, j);
    float v = Xsig_pred_(2, j);
    float psi = Xsig_pred_(3, j);

    float ro = sqrt(px*px + py*py);
    if (ro < 0.001) {
      ro = 0.001;
    }
    float phi = atan2(py, px);
    float ro_dot = v*(px*cos(psi) + py*sin(psi))/ro;

    Zsig_.col(j) << ro, phi, ro_dot;
  }

  z_pred_.fill(0.0);
  for (int j = 0; j < n_sigma_; ++j) {
    z_pred_ += weights_(j)*Zsig_.col(j);
  }

  S_.fill(0.0);
  S_ += R_radar_;

  for (int j = 0; j < n_sigma_; ++j) {
    VectorXd diff = Zsig_.col(j) - z_pred_;
    diff(1) = Tools::NormalizeAngle(diff(1));
    S_ += weights_(j)*diff*diff.transpose();
  }
}

void UKF::UpdateState(VectorXd z) {
  MatrixXd Tc = MatrixXd(n_x_, n_z_);
  Tc.fill(0.0);

  for (int j = 0; j < n_sigma_; ++j) {
    VectorXd diff_x = Xsig_pred_.col(j) - x_;
    diff_x(3) = Tools::NormalizeAngle(diff_x(3));

    VectorXd diff_z = Zsig_.col(j) - z_pred_;
    diff_z(1) = Tools::NormalizeAngle(diff_z(1));

    Tc += weights_(j)*diff_x*diff_z.transpose();
  }

  MatrixXd K = Tc * S_.inverse();

  VectorXd diff = z - z_pred_;
  diff(1) = Tools::NormalizeAngle(diff(1));

  x_ += K * diff;
  P_ -= K * S_ * K.transpose();
}

#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

  VectorXd rmse = VectorXd::Zero(4);

  if (estimations.size() == 0
      || estimations.size() != ground_truth.size()) {
    return rmse;
  }

  for(int i=0; i < estimations.size(); ++i) {
    VectorXd c = estimations[i] - ground_truth[i];
    VectorXd c_2 = c.array()*c.array();
    rmse += c_2;
  }

  rmse /= estimations.size();

  return rmse.array().sqrt();

}

double Tools::NormalizeAngle(double angle) {
  return atan2(sin(angle), cos(angle));
}

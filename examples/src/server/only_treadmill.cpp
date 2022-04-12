//
// Created by Jemin Hwangbo on 2/28/19.
// MIT License
//
// Copyright (c) 2019-2019 Robotic Systems Lab, ETH Zurich
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


//#include <raisim/OgreVis.hpp>
//#include "raisimBasicImguiPanel.hpp"
#include "helper.hpp"
//#include "Quaternion.h"
//#include "Slerp.h"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
//#include <stdio.h>
#include <GL/glut.h>
#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"

using namespace std;
using json = nlohmann::json;


int gcDim_, gvDim_, nJoints_;
bool visualizable_ = false;
//std::normal_distribution<double> distribution_;
raisim::ArticulatedSystem *anymal_, *anymal_ref_;
raisim::Ground *ground_;
//std::vector<GraphicObject> *anymalVisual_;
//std::vector<GraphicObject> *anymalVisual_ref;
Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget28_, vTarget_, torque_;
Eigen::VectorXd gc_ref_, gv_ref_;

// mingi
std::vector<Eigen::VectorXd> ref_motion_;
std::string ref_motion_file_;
std::vector<double> cumultime_;
double time_ref_motion_start_ = 0.;
double time_episode_ = 0;

double t = 0;
double dt = 0.001;
double control_dt_ = 0.002;

Eigen::VectorXd ref_motion_one(37);
Eigen::VectorXd gait_motion_one(39);

raisim::TerrainProperties terrainProperties;

raisim::Box *box;

double setpoint;
double input;
double prev_input;
double kp;
double kd;
double output;

Eigen::VectorXd box_gc_, box_gv_;

nlohmann::json outputs;
int json_count_ = 0;

void stdPID(double &setpoint,
            double &input,
            double &prev_input,
            double &kp,
            double &kd,
            double &output
){
  double error;
  double dInput;
  double pterm, dterm;

  error = setpoint - input; //오차 = 설정값 - 현재 입력값
  dInput = input - prev_input;
  prev_input = input; //다음 주기에 사용하기 위해서 현재 입력값을 저장//

  //PID제어//
  pterm = kp * error; //비례항
  dterm = -kd * dInput / dt; //미분항(미분항은 외력에 의한 변경이므로 setpoint에 의한 내부적인 요소를 제외해야 한다.(-) 추가)//

  output = pterm + dterm; //Output값으로 PID요소를 합한다.//
}


int main(int argc, char **argv) {

  raisim::World::setActivationKey("/home/opensim2020/raisim_v2_workspace/activation_MINGI JUNG_KAIST.raisim");

  /// create raisim world
  raisim::World world;

  world.setTimeStep(0.001);

  auto fps = 50.0;

  /// create raisim objects

  //auto ground = world.addHeightMap("/home/opensim2020/raisim_v2_workspace/raisimlib/examples/src/server/KAIST_terrain/straight_80cm.png", 34, 1.24, 80, 80, 0.00027, -8.848); //box narrow object
  auto ground = world.addHeightMap("/home/opensim2020/raisim_v2_workspace/raisimlib/examples/src/server/KAIST_terrain/straight_80cm.png", 34, 1.24, 80, 80, 0.00027, -8.948); //box narrow object



  raisim::RaisimServer server(&world);
  server.launchServer(8080);
  //server.focusOn(box);

  // box = world.addBox(80, 0.8, 0.001, 99999);
  // box->setPosition(34, 1.20, -0.032);

  // box = world.addBox(20, 2, 0.0001, 99999);
  //
  // box->setBodyType(raisim::BodyType::KINEMATIC);
  //
  // box->setPosition(8, 1.20, -0.033);
  //
  // box->setVelocity(-1, 0, 0, 0, 0, 0);

  box = world.addBox(20, 0.8, 0.1, 1000);

  box->setBodyType(raisim::BodyType::DYNAMIC);

  box->setPosition(8, 1.20, -0.086);

  //box->setVelocity(-1, 0, 0, 0, 0, 0);

  box->setAppearance("green");

  raisim::Vec<3> ext_force_;
  //ext_force_[0] = 1043;
  ext_force_[0] = 0;
  ext_force_[1] = 0;
  ext_force_[2] = 0;

  box_gc_.setZero(7);
  box_gv_.setZero(6);

  Eigen::Vector3d box_vel_;

  // setpoint = -1.0789;
  setpoint = -4.2637;
  input = 0;
  prev_input = 0;
  // kp = 1;
  // kd = 0.1;
  kp = 100;
  kd = 0.1;
  output = 0;

  int control_count=0;

  for (int i=0; i<200000000; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000));

    if (control_count==9){
      box_vel_ = box->getLinearVelocity();
      input = (double) box_vel_[0];
      stdPID(setpoint, input, prev_input, kp, kd, output);

      ext_force_[0] += 500 * output;
      //ext_force_[0] += 1 * output * abs(output);

      // if(ext_force_[0] > 2000){
      //   ext_force_[0] = 2000;
      // }
      // if(ext_force_[0] < -2000){
      //   ext_force_[0] = -2000;
      // }

      control_count = 0;
    }

    control_count += 1;
    box->setExternalForce(0, ext_force_);
    box_vel_ = box->getLinearVelocity();

    // outputs["box_vel"][json_count_] = box_vel_[0];
    // outputs["ext_force"][json_count_] = ext_force_[0];
    //
    // std::ofstream o("treadmill_vel_0_01_1500.json");
    // o << std::setw(4) << outputs << std::endl;
    // json_count_ += 1;

    // cout<<box_vel_[0]<<endl;
    //box->setPosition(8, 1.20, -0.083);
    // cout<<"output: "<<output<<endl;
    cout<<"ext_force_[0]: "<<ext_force_[0]<<endl;

    // auto box_quat = box->getQuaternion();
    // cout<<"box_quat[0]: "<<box_quat[0]<<endl;
    // cout<<"box_quat[1]: "<<box_quat[1]<<endl;
    // cout<<"box_quat[2]: "<<box_quat[2]<<endl;
    // cout<<"box_quat[3]: "<<box_quat[3]<<endl;
    server.integrateWorldThreadSafe();
  }

  // for (int i=0; i<200000000; i++) {
  //   std::this_thread::sleep_for(std::chrono::microseconds(1000));
  //
  //   if (i >4000){
  //     ext_force_[0] = 0;
  //   }
  //   // if (control_count==9){
  //   //   box_vel_ = box->getLinearVelocity();
  //   //   input = (double) box_vel_[0];
  //   //   stdPID(setpoint, input, prev_input, kp, kd, output);
  //   //
  //   //   ext_force_[0] += 500 * output;
  //   //
  //   //   control_count = 0;
  //   // }
  //   //
  //   // control_count += 1;
  //   box->setExternalForce(0, ext_force_);
  //   box_vel_ = box->getLinearVelocity();
  //   cout<<"box_vel_[0]: "<<box_vel_[0]<<endl;
  //   // cout<<"output: "<<output<<endl;
  //   // cout<<"ext_force_[0]: "<<ext_force_[0]<<endl;
  //   server.integrateWorldThreadSafe();
  // }

  server.killServer();

}

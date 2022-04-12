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



class Quaternion
{
public:
    Quaternion(double wVal, double xVal, double yVal, double zVal) {
        w = wVal; x= xVal; y = yVal; z = zVal;
    }

    Quaternion() {
        Quaternion(1, 0, 0, 0);
    }

    double getW() { return w; }
    double getX() { return x; }
    double getY() { return y; }
    double getZ() { return z; }
    void set(double wVal, double xVal, double yVal, double zVal) {
        w = wVal; x= xVal; y = yVal; z = zVal;
    }

private:
    double w, x, y, z;
};

void slerp(Quaternion &qout, Quaternion q1, Quaternion q2, double lambda)
{
    double w1, x1, y1, z1, w2, x2, y2, z2;
    double theta, mult1, mult2;

    w1 = q1.getW(); x1 = q1.getX(); y1 = q1.getY(); z1 = q1.getZ();
    w2 = q2.getW(); x2 = q2.getX(); y2 = q2.getY(); z2 = q2.getZ();

    // Reverse the sign of q2 if q1.q2 < 0.
    if (w1*w2 + x1*x2 + y1*y2 + z1*z2 < 0)
    {
        w2 = -w2; x2 = -x2; y2 = -y2; z2 = -z2;
    }

    theta = acos(w1*w2 + x1*x2 + y1*y2 + z1*z2);

    if (theta > 0.000001)
    {
        mult1 = sin( (1-lambda)*theta ) / sin( theta );
        mult2 = sin( lambda*theta ) / sin( theta );
        // To avoid division by 0 and by very small numbers the approximation of sin(angle)
        // by angle is used when theta is small (0.000001 is chosen arbitrarily).
    }
    else
    {
        mult1 = 1 - lambda;
        mult2 = lambda;
    }

    double w3 =  mult1*w1 + mult2*w2;
    double x3 =  mult1*x1 + mult2*x2;
    double y3 =  mult1*y1 + mult2*y2;
    double z3 =  mult1*z1 + mult2*z2;

    qout.set(w3, x3, y3, z3);
}


void quatToEulerVec(const double* quat, double* eulerVec) {
  const double norm = (std::sqrt(quat[1]*quat[1] + quat[2]*quat[2] + quat[3]*quat[3]));
  if(fabs(norm) < 1e-12) {
    eulerVec[0] = 0;
    eulerVec[1] = 0;
    eulerVec[2] = 0;
    return;
  }

  const double normInv = 1.0/norm;
  const double angleNomrInv = std::acos(std::min(quat[0],1.0))*2.0*normInv;
  eulerVec[0] = quat[1] * angleNomrInv;
  eulerVec[1] = quat[2] * angleNomrInv;
  eulerVec[2] = quat[3] * angleNomrInv;
}




// void setupCallback() {
//   auto vis = raisim::OgreVis::get();
//
//   /// light
//   vis->getLight()->setDiffuseColour(1, 1, 1);
//   vis->getLight()->setCastShadows(true);
//   vis->getLightNode()->setPosition(3, 3, 3);
//
//   /// load textures
//   vis->addResourceDirectory(vis->getResourceDir() + "/material/checkerboard");
//   vis->loadMaterialFile("checkerboard.material");
//
//   vis->addResourceDirectory(vis->getResourceDir() + "/material/skybox/violentdays");
//   vis->loadMaterialFile("violentdays.material");
//
//   /// shdow setting
//   vis->getSceneManager()->setShadowTechnique(Ogre::SHADOWTYPE_TEXTURE_ADDITIVE);
//   vis->getSceneManager()->setShadowTextureSettings(2048, 3);
//
//   /// scale related settings!! Please adapt it depending on your map size
//   // beyond this distance, shadow disappears
//   vis->getSceneManager()->setShadowFarDistance(10);
//   // size of contact points and contact forces
//   vis->setContactVisObjectSize(0.03, 0.2);
//   // speed of camera motion in freelook mode
//   vis->getCameraMan()->setTopSpeed(5);
//
//   /// skyboxefined reference to `Quaternion<double>::Quaternion(double, double, double, double)'
//
//   Ogre::Quaternion quat;
//   quat.FromAngleAxis(Ogre::Radian(M_PI_2), {1., 0, 0});
//   vis->getSceneManager()->setSkyBox(true,
//                                     "Examples/StormySkyBox", //"Examples/StormySkyBox" "skybox/violentdays"
//                                     500,
//                                     true,
//                                     quat,
//                                     Ogre::ResourceGroupManager::AUTODETECT_RESOURCE_GROUP_NAME);
// }


void read_ref_motion() {
    Eigen::VectorXd jointNominalConfig(37);
    double totaltime = 0;
    std::ifstream infile1("/home/opensim2020/raisim_v3_workspace/raisimLib/examples/src/server/motions/mingi_run_v2.txt");

    nlohmann::json  jsondata;
    infile1 >> jsondata;

    cumultime_.push_back(0);

    Eigen::VectorXd jointIdxConvTable(37);
    jointIdxConvTable <<
        // 0, 2, 1,          // 1, 2, 3         // root translation
        // 3, 4, 5, 6,       // 4, 5, 6, 7      // root rotation
        // 7, 8, 9, 10,      // 8, 9, 10, 11    // lumbar
        // 11, 12, 13, 14,   // 12, 13, 14, 15  // neck
        // 24, 25, 26, 27,   // 16, 17, 18, 19  // right shoulder
        // 28,               // 20              // right elbow
        // 29, 30, 31, 32,   // 21, 22, 23, 24  // left shoulder
        // 15,               // 25              // left elbow
        // 16, 17, 18, 19,   // 26, 27, 28, 29  // right hip
        // 33,               // 30              // right knee
        // 34, 35, 36,   // 31, 32, 33, 34  // right ankle
        // 37, 38, 39, 40,   // 35, 36, 37, 38, // left hip
        // 20,               // 39              // left knee
        // 21, 22, 23;   // 40, 41, 42, 43  // left ankle

        0, 2, 1,          // root translation
        3, 4, 5, 6,       // root rotation

        7, 8, 9, 10,      // lumbar

        21, 22, 23, 24,   // right hip
        25,               // right knee
        26, 27, 28,       // right ankle

        11, 12, 13, 14,   // right shoulder
        15,               // right elbow

        29, 30, 31, 32,   // left hip
        33,               // left knee
        34, 35, 36,       // left ankle

        16, 17, 18, 19,   // left shoulder
        20;               // left elbow


    int nframe = jsondata["Frames"].size();
    for (int iframe = 0; iframe < nframe; iframe++) {
        int ncol = jsondata["Frames"][iframe].size();
        //assert( ncol== 44); // idx zero is time

        jointNominalConfig.setZero();
        for (int icol = 1; icol < ncol; icol++)
            jointNominalConfig[jointIdxConvTable[icol-1]] = jsondata["Frames"][iframe][icol];

        // x direction 90 degree rotation_start
        double qw1 = jointNominalConfig[3];
        Eigen::Vector3d qv1(jointNominalConfig[4], jointNominalConfig[5], jointNominalConfig[6]);
        double qw2 = 0.7071068;
        Eigen::Vector3d qv2(0.7071068, 0, 0);
        // double qw2 = 1;
        // Eigen::Vector3d qv2(0, 0, 0);

        double qw3;
        Eigen::Vector3d qv3(0, 0, 0);
        double qw4 = 0;
        Eigen::Vector3d qv4(0, 0, 1);
        double qw5;
        Eigen::Vector3d qv5(0, 0, 0);

        qw3 = qw2 * qw1 - qv2.dot(qv1);
        qv3 = qw2 * qv1 + qw1 * qv2 + qv2.cross(qv1);

        qw5 = qw4 * qw3 - qv4.dot(qv3);
        qv5 = qw4 * qv3 + qw3 * qv4 + qv4.cross(qv3);

        jointNominalConfig[3] = qw3;
        jointNominalConfig[4] = qv3[0];
        jointNominalConfig[5] = qv3[1];
        jointNominalConfig[6] = qv3[2];
        // x direction 90 degree rotation_end

        // jointNominalConfig[3] = qw5;
        // jointNominalConfig[4] = qv5[0];
        // jointNominalConfig[5] = qv5[1];
        // jointNominalConfig[6] = qv5[2];

        //x_translation add
        //jointNominalConfig[0] += 1.6;

        ref_motion_.push_back(jointNominalConfig);

        totaltime += (double) jsondata["Frames"][iframe][0];
        cumultime_.push_back(totaltime);
    }

    cumultime_.pop_back();
    //cout<<cumultime_.size()<<endl;
    //cout<<cumultime_[40]<<endl;
}


void calc_slerp_joint(
        Eigen::VectorXd &ref_motion_one,                // output
        const int *idxlist,                             // indices to be updated
        std::vector<Eigen::VectorXd> &ref_motion,       // input
        int idxprev,                                    // input
        int idxnext,                                    // input
        double t_offset_ratio) {                        // input

    int idxW = idxlist[0];
    int idxX = idxlist[1];
    int idxY = idxlist[2];
    int idxZ = idxlist[3];

    Quaternion qprev(ref_motion_[idxprev][idxW], ref_motion_[idxprev][idxX],
                     ref_motion_[idxprev][idxY], ref_motion_[idxprev][idxZ]);
    Quaternion qnext(ref_motion_[idxnext][idxW], ref_motion_[idxnext][idxX],
                     ref_motion_[idxnext][idxY], ref_motion_[idxnext][idxZ]);
    Quaternion qout;
    slerp(qout, qprev, qnext, t_offset_ratio);
    ref_motion_one[idxW] = qout.getW();
    ref_motion_one[idxX] = qout.getX();
    ref_motion_one[idxY] = qout.getY();
    ref_motion_one[idxZ] = qout.getZ();
}

void calc_interp_joint(
        Eigen::VectorXd &ref_motion_one,                // output
        int idxjoint,                                   // index of a scalar joint to be updated
        std::vector<Eigen::VectorXd> &ref_motion,       // input
        int idxprev,                                    // input
        int idxnext,                                    // input
        double t_offset_ratio) {                        // input

    double a = ref_motion_[idxprev][idxjoint];
    double b = ref_motion_[idxnext][idxjoint];

    ref_motion_one[idxjoint] = a + (b - a) * t_offset_ratio;
}

void get_ref_motion(Eigen::VectorXd &ref_motion_one, double tau) {
    int idx_frame_prev = -1;
    int idx_frame_next = -1;
    double t_offset;
    double t_offset_ratio;

    // SKOO 20200629 This loop should be checked again.
    for (int i = 1; i < cumultime_.size(); i++) { // 108 means (the number of cumultime + 1)
        if (tau < cumultime_[i]) {
            idx_frame_prev = i - 1; // this index is including 0
            idx_frame_next = i;
            break;
        }
    }

    t_offset = tau - cumultime_[idx_frame_prev];
    t_offset_ratio = t_offset / (cumultime_[idx_frame_next] - cumultime_[idx_frame_prev]);

    int idx_rootx         = 0;
    int idx_rooty         = 1;
    int idx_rootz         = 2;
    int idx_qroot[4]      = {3, 4, 5, 6};
    int idx_qlumbar[4]    = {7, 8, 9, 10};
    int idx_qrshoulder[4] = {11, 12, 13, 14};
    int idx_relbow        = 15;
    int idx_qlshoulder[4] = {16, 17, 18, 19};
    int idx_lelbow        = 20;
    int idx_qrhip[4]      = {21, 22, 23, 24};
    int idx_rknee         = 25;
    int idx_rankle        = 26;
    int idx_rsubtalar     = 27;
    int idx_rmtp          = 28;
    int idx_qlhip[4]      = {29, 30, 31, 32};
    int idx_lknee         = 33;
    int idx_lankle        = 34;
    int idx_lsubtalar     = 35;
    int idx_lmtp          = 36;

    calc_slerp_joint(ref_motion_one, idx_qroot, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_slerp_joint(ref_motion_one, idx_qlumbar, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_slerp_joint(ref_motion_one, idx_qrshoulder, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_slerp_joint(ref_motion_one, idx_qlshoulder, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_slerp_joint(ref_motion_one, idx_qrhip, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_slerp_joint(ref_motion_one, idx_qlhip, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);

    calc_interp_joint(ref_motion_one, idx_rootx, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_rooty, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_rootz, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_relbow, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_lelbow, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_rknee, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_rankle, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_rsubtalar, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_rmtp, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_lknee, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_lankle, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_lsubtalar, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    calc_interp_joint(ref_motion_one, idx_lmtp, ref_motion_, idx_frame_prev, idx_frame_next, t_offset_ratio);
}

void get_gv_init(
        Eigen::VectorXd &gv_init_, //output
        double tau){

          double duration_ref_motion = cumultime_.back(); // in case, tau - control_dt is minus

          Eigen::VectorXd ref_motion_one_next(37);
          Eigen::VectorXd ref_motion_one_prev(37); //mingi, to calculate velocity by 미분


          double small_dt = 0.004;

          if (tau + small_dt/2 <= duration_ref_motion){
              get_ref_motion(ref_motion_one_next, tau + small_dt/2);
          }
          else{ // if tau - control_dt is minus
              get_ref_motion(ref_motion_one_next, tau + small_dt/2 - duration_ref_motion);
          }

          if (tau - small_dt/2 >= 0){
              get_ref_motion(ref_motion_one_prev, tau - small_dt/2);
          }
          else{ // if tau - control_dt is minus
              get_ref_motion(ref_motion_one_prev, duration_ref_motion + tau - small_dt/2);
          }

          Eigen::VectorXd jointIdx_quat_ref(6);
          Eigen::VectorXd jointIdx_euler_obs(6);

          jointIdx_quat_ref << //quaternion w idx
              3,
              7, 11,        // lumbar, right shoulder
              16, 21,       // left shoulder, right hip
              29;    // left hip
          jointIdx_euler_obs << //EULER idx
              3,
              6, 9,       // lumbar, right shoulder
              13, 17,       // left shoulder, right hip
              24;    // left hip

          for (int j = 0; j < 37; j++){
            for (int i = 0; i < 6; i++){ // jointIdx_quat_ref size
              if (j == jointIdx_quat_ref[i]){
                double qw1 = ref_motion_one_next[j];
                Eigen::Vector3d qv1(ref_motion_one_next[j + 1],ref_motion_one_next[j + 2], ref_motion_one_next[j + 3]);
                double qw2 = ref_motion_one_prev[j];
                Eigen::Vector3d qv2(ref_motion_one_prev[j + 1], ref_motion_one_prev[j + 2], ref_motion_one_prev[j + 3]);
                double qw3;
                Eigen::Vector3d qv3;//(0, 0, 0);
                double qw4;
                Eigen::Vector3d qv4;//(0, 0, 0);

                qw3 = qw1 / (qw1 * qw1 + qv1[0] * qv1[0] + qv1[1] * qv1[1] + qv1[2] * qv1[2]); //inverse
                qv3[0] = -1 * qv1[0] / (qw1 * qw1 + qv1[0] * qv1[0] + qv1[1] * qv1[1] + qv1[2] * qv1[2]);
                qv3[1] = -1 * qv1[1] / (qw1 * qw1 + qv1[0] * qv1[0] + qv1[1] * qv1[1] + qv1[2] * qv1[2]);
                qv3[2] = -1 * qv1[2] / (qw1 * qw1 + qv1[0] * qv1[0] + qv1[1] * qv1[1] + qv1[2] * qv1[2]);

                qw4 = qw2 * qw3 - qv2.dot(qv3); //between q2, q3
                qv4 = qw2 * qv3 + qw3 * qv2 + qv2.cross(qv3);

                // qw4 = qw3 * qw2 - qv3.dot(qv2); //between q2, q3
                // qv4 = qw3 * qv2 + qw2 * qv3 + qv3.cross(qv2);


                // double qw5;
                // Eigen::Vector3d qv5;//(0, 0, 0);
                //
                // qw5 = (qw1 - qw2)/(small_dt);
                // qv5[0] = (qv1[0] - qv2[0])/(small_dt);
                // qv5[1] = (qv1[1] - qv2[1])/(small_dt);
                // qv5[2] = (qv1[2] - qv2[2])/(small_dt);
                //
                // double qw6;
                // Eigen::Vector3d qv6;//(0, 0, 0);
                //
                // qw6 = qw4 / (qw4 * qw4 + qv4[0] * qv4[0] + qv4[1] * qv4[1] + qv4[2] * qv4[2]); //inverse
                // qv6[0] = -1 * qv4[0] / (qw4 * qw4 + qv4[0] * qv4[0] + qv4[1] * qv4[1] + qv4[2] * qv4[2]);
                // qv6[1] = -1 * qv4[1] / (qw4 * qw4 + qv4[0] * qv4[0] + qv4[1] * qv4[1] + qv4[2] * qv4[2]);
                // qv6[2] = -1 * qv4[2] / (qw4 * qw4 + qv4[0] * qv4[0] + qv4[1] * qv4[1] + qv4[2] * qv4[2]);
                //
                // double qw7;
                // Eigen::Vector3d qv7;//(0, 0, 0);
                //
                // qw7 = qw5 * qw6 - qv5.dot(qv6); //between q2, q3
                // qv7 = qw5 * qv6 + qw6 * qv5 + qv5.cross(qv6);
                //
                // gv_init_[jointIdx_euler_obs[i]] = 2*qv7[0];
                // gv_init_[jointIdx_euler_obs[i]+1] = 2*qv7[1];
                // gv_init_[jointIdx_euler_obs[i]+2] = 2*qv7[2];

                //mingi method6 start using theta

                double theta;

                if (qw4 > 0.99) {
                    theta = acos(0.99) * 2;
                }
                else {
                    theta = acos(qw4) * 2;
                }

                double vv1, vv2, vv3;

                vv1 = qv4[0] / sin(theta/2);
                vv2 = qv4[1] / sin(theta/2);
                vv3 = qv4[2] / sin(theta/2);

                gv_init_[jointIdx_euler_obs[i]] = -theta*vv1/(2*control_dt_);
                gv_init_[jointIdx_euler_obs[i]+1] = -theta*vv2/(2*control_dt_);
                gv_init_[jointIdx_euler_obs[i]+2] = -theta*vv3/(2*control_dt_);

                //mingi method6 end



              }
            }

            if (j == 0 || j == 1 || j == 2 ||j == 15 || j == 20 || j == 25 || j == 26 || j == 27 || j == 28 || j == 33 || j == 34 || j == 35 || j == 36){ //rev joints
                if (j == 0){
                    gv_init_[0] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 1){
                    gv_init_[1] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 2){
                    gv_init_[2] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 15) {
                    gv_init_[12] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 20) {
                    gv_init_[16] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 25) {
                    gv_init_[20] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 26) {
                    gv_init_[21] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 27) {
                    gv_init_[22] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 28) {
                    gv_init_[23] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 33) {
                    gv_init_[27] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 34) {
                    gv_init_[28] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 35) {
                    gv_init_[29] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
                if (j == 36) {
                    gv_init_[30] = (ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt;
                }
            }
          }

    // //mingi makes z-axis angular velocity zero
    // gv_init_[7] = 0;
    // gv_init_[10] = 0;
    // gv_init_[13] = 0;
    // gv_init_[17] = 0;
    // gv_init_[21] = 0;
    // gv_init_[25] = 0;
    // gv_init_[28] = 0;
    // gv_init_[32] = 0;


}

void get_gait_motion(Eigen::VectorXd &ref_motion_one, Eigen::VectorXd &gait_motion_one) {
  for (int i=0; i < 39; i++){
    if (i <11){
      gait_motion_one[i] = ref_motion_one[i];
    }
    else{
      gait_motion_one[i] = ref_motion_one[i+4];
    }
  }
}



int main(int argc, char **argv) {

  raisim::World::setActivationKey("/home/opensim2020/raisim_v2_workspace/activation_MINGI JUNG_KAIST.raisim");

  /// create raisim world
  raisim::World world;

  world.setTimeStep(0.001);

  auto fps = 50.0;
  /// these method must be called before initApp

  /// starts visualizer thread
  //vis->initApp();

  // // mingi random_map
  // terrainProperties.frequency = 0.2;
  // terrainProperties.zScale = 1.5;
  // terrainProperties.xSize = 60.0;
  // terrainProperties.ySize = 60.0;
  // terrainProperties.xSamples = 80;
  // terrainProperties.ySamples = 80;
  // terrainProperties.fractalOctaves = 3;
  // terrainProperties.fractalLacunarity = 0.2;
  // terrainProperties.fractalGain = 0.025;


  /// create raisim objects
  //auto ground = world.addHeightMap(0.0, 0.0, terrainProperties);
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/mingis.png"),0, 0, 100, 100, 0.0007, -1.302);
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/KAIST Height Map (Merged).png"),0, 0, 100, 100, 0.0002, -0.882);
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/KAIST_goal.png"),0, 0, 100, 100, 0.0002, -0.882);
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/KAIST Height Map (Merged).png"),0, 0, 100, 100, 0.00007, -0.302);
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/mingi1081.png"),28, 22.5, 60, 60, 0.0002, -6.522);
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/ground.png"),0, 0, 60, 60, 0.0002, -6.522);
  //auto ground = world.addHeightMap("/home/opensim2020/raisim_v3_workspace/raisimLib/examples/src/server/KAIST_terrain/ground.png",0, 0, 60, 60, 0.0002, -6.560); //mingi_walk
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/ground.png"),0, 0, 60, 60, 0.0002, -6.562); //mingi_run
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/terrain_slope_descend.png"), 30, 0, 80, 80, 0.00027, -10.30); //mingi_slope_descend
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/3_mingi.png"),34, 1.28, 80, 80, 0.00027, -8.848); // mid_among 5
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/3_mingi.png"),34, 19.00, 80, 80, 0.00027, -8.848); // descend_long_among 5
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/3_mingi.png"),34, -19.49, 80, 80, 0.00027, -8.848); // ascend_long_among 5
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/3_mingi.png"),34, -9.105, 80, 80, 0.00027, -8.848); // ascend_short_among 5
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/3_mingi.png"),34, 10.14, 80, 80, 0.00027, -8.848); // descend_short_among 5
  //auto ground = world.addHeightMap(raisim::loadResource("KAIST_terrain/terrain_slope_ascend.png"), 28, 0, 80, 80, 0.00027, -9.41); //mingi_slope_ascend
  //auto ground = world.addHeightMap("/home/opensim2020/raisim_v3_workspace/raisimLib/examples/src/server/KAIST_terrain/terrain_stairs_ascend.png", 28.2, 0, 80, 80, 0.00027, -9.24); //mingi_stairs_ascend
  //auto ground = world.addHeightMap("/home/opensim2020/raisim_v3_workspace/raisimLib/examples/src/server/KAIST_terrain/terrain_stairs_ascend.png", 28.2, 0, 80, 80, 0.00027, -9.30); //exp
  //auto ground = world.addHeightMap("/home/opensim2020/raisim_v3_workspace/raisimLib/examples/src/server/KAIST_terrain/terrain_stairs_descend.png", 28.1, 0, 80, 80, 0.00027, -7.83); //mingi_stairs_descend
  //auto ground = world.addHeightMap("/home/opensim2020/raisim_v2_workspace/raisimlib/examples/src/server/KAIST_terrain/straight_only_2.png", 34, 1.24, 80, 80, 0.00027, -8.848); //box narrow object
  //auto ground = world.addHeightMap("/home/opensim2020/raisim_v2_workspace/raisimlib/examples/src/server/KAIST_terrain/straight_deep_only.png", 34, 1.24, 80, 80, 0.00027, -8.848); //box narrow object
  //auto ground = world.addHeightMap("/home/opensim2020/raisim_v2_workspace/raisimlib/examples/src/server/KAIST_terrain/straight_80cm.png", 34, 1.24, 80, 80, 0.00027, -8.848); //box narrow object
  auto ground = world.addHeightMap("/home/opensim2020/raisim_v2_workspace/raisimlib/examples/src/server/KAIST_terrain/straight_80cm.png", 34, 1.24, 80, 80, 0.00027, -8.948); //box narrow object
  //auto ground = world.addHeightMap("/home/opensim2020/raisim_v2_workspace/raisimlib/examples/src/server/KAIST_terrain/slope_up_down.png", 28, 0, 80, 80, 0.00027, -9.41); //mingi_slope_up_down

  auto robot = world.addArticulatedSystem("/home/opensim2020/raisim_v2_workspace/raisimlib/examples/src/server/full_body/MSK_GAIT2392_model_20201117.urdf");
  robot->setName("bullet_humanoid");
  auto order = robot->getMovableJointNames();

  raisim::RaisimServer server(&world);
  server.launchServer(8081);
  server.focusOn(robot);

  // joint order

  for (int i = 0; i < order.size(); i++){
    cout<<order[i]<<endl;
  }


  // raisim::Vec<3> ankle_r_position;
  // robot->getFramePosition("walker_knee_r", ankle_r_position);
  //
  // cout<<ankle_r_position[0]<<endl;
  // cout<<ankle_r_position[1]<<endl;
  // cout<<ankle_r_position[2]<<endl;



  gcDim_ = robot->getGeneralizedCoordinateDim();
  gvDim_ = robot->getDOF();
  gc_.setZero(gcDim_);
  gv_.setZero(gvDim_);
  //cout<<gvDim_<<endl;

  //gc_init_.setZero(39);

  // gc_init_ <<
  //         0, 0, 1, //Pelvis translation
  //         0.707, 0.707, 0, 0, //pelvis orientation
  //         1, 0, 0, 0, //back
  //         1, 0, 0, 0, //shoulder_r
  //         0, //elbow_r
  //         1, 0, 0, 0, //shoulder_l
  //         0, //elbow_l
  //         1, 0, 0, 0, //hip_r
  //         0, //walker_knee_r
  //         1, 0, 0, 0, //ankle_r
  //         1, 0, 0, 0, //hip_l
  //         0, //walker_knee_l
  //         1, 0, 0, 0; //ankle_l

  double map_height;
  map_height = ground->getHeight(0,0);

  cout<<map_height<<endl;
  // Eigen::Vector3d mingi(3, 4, 5);
  //
  // mingi.pop_back();
  //
  // cout<<mingi.size()<<endl;

  //robot->setGeneralizedCoordinate(gc_init_);

  gv_init_.setZero(31);

  read_ref_motion();

  double duration_ref_motion = cumultime_.back();
  cout<<duration_ref_motion<<endl;

  // // random start
  //
  // std::random_device rd; //random_device 생성
  // std::mt19937 gen(rd()); //난수 생성 엔진 초기화
  // std::uniform_int_distribution<int> dis(0,99); //0~99 까지 균등하게 나타나는 난수열 생성
  //
  // time_episode_ = ((double) dis(gen)/100) * duration_ref_motion;
  //
  // // random end



  int n_walk_cycle = 0;
  double tau = 0;
  n_walk_cycle = (int) (time_episode_ / duration_ref_motion);
  tau = time_episode_ - n_walk_cycle * duration_ref_motion;

  //tau = 0.5 * duration_ref_motion;
  get_ref_motion(ref_motion_one, tau);

  robot->setGeneralizedForce(Eigen::VectorXd::Zero(robot->getDOF()));
  robot->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);

  //ref_motion_one[2] += 0.3;
  //robot->setGeneralizedCoordinate(ref_motion_one);

  //get_gait_motion(ref_motion_one, gait_motion_one);
  ref_motion_one[1] += 1.33;
  robot->setGeneralizedCoordinate(ref_motion_one);

  // box = world.addBox(80, 0.8, 0.001, 99999);
  // box->setPosition(34, 1.20, -0.032);

  // box = world.addBox(20, 2, 0.0001, 99999);
  //
  // box->setBodyType(raisim::BodyType::KINEMATIC);
  //
  // box->setPosition(8, 1.20, -0.033);
  //
  // box->setVelocity(-1, 0, 0, 0, 0, 0);

  // box = world.addBox(20, 0.8, 0.1, 99999);
  box = world.addBox(40, 0.8, 0.1, 99999);

  box->setBodyType(raisim::BodyType::DYNAMIC);

  // box->setPosition(8, 1.20, -0.083);
  box->setPosition(17, 1.20, -0.083);

  box->setVelocity(-1, 0, 0, 0, 0, 0);



  raisim::Vec<3> ankle_r_position;
  robot->getFramePosition("ankle_l", ankle_r_position);

  cout<<ankle_r_position[0]<<endl;
  cout<<ankle_r_position[1]<<endl;
  cout<<ankle_r_position[2]<<endl;



  // // direction check
  // //robot->getGeneralizedCoordinate(gc_);
  // raisim::Vec<4> quat;
  // raisim::Mat<3, 3> rot;
  // double obdouble_a, obdouble_b, obdouble_c;
  // quat[0] = ref_motion_one[3];
  // quat[1] = ref_motion_one[4];
  // quat[2] = ref_motion_one[5];
  // quat[3] = ref_motion_one[6];
  // raisim::quatToRotMat(quat, rot);
  // obdouble_a = rot.e().row(2)[0];
  // obdouble_b = rot.e().row(2)[1];
  // obdouble_c = rot.e().row(2)[2];
  //
  // cout<<"obdouble_a: "<<obdouble_a<<endl;
  // cout<<"obdouble_b: "<<obdouble_b<<endl;
  // cout<<"obdouble_c: "<<obdouble_c<<endl;
  // cout<<ref_motion_one[3]<<endl;
  // cout<<ref_motion_one[4]<<endl;
  // cout<<ref_motion_one[5]<<endl;
  // cout<<ref_motion_one[6]<<endl;


  // Eigen::VectorXd gf_;
  // gf_.setZero(gvDim_);
  //gf_ = robot -> getGeneralizedForce();
  // auto gf_ = robot -> getGeneralizedForce();
  // cout<<gf_<<endl;

  // Height map experiment end

  // double map_height;
  // map_height = ground->getHeight(0,0);
  //
  // cout<<map_height<<endl;

  //robot->getState(gc_, gv_);

  //cout<<gc_[2]<<endl;
  //cout<<gc_[3]<<gc_[4]<<gc_[5]<<gc_[6]<<endl;

  //cout<<atan2(-3,2)<<endl;


  // double qw1 = gc_[3];
  // Eigen::Vector3d qv1(gc_[4], gc_[5], gc_[6]);
  // double qw2 = 0;
  // Eigen::Vector3d qv2(0, 0.3593, 0.9332);
  // double qw3;
  // Eigen::Vector3d qv3;//(0, 0, 0);
  //
  // qw3 = qw2 * qw1 - qv2.dot(qv1); //between q2, q3
  // qv3 = qw2 * qv1 + qw1 * qv2 + qv2.cross(qv1);
  //
  // gc_[3] = qw3;
  // gc_[4] = qv3[0];
  // gc_[5] = qv3[1];
  // gc_[6] = qv3[2];
  //
  //
  // raisim::Vec<4> quat;
  // raisim::Mat<3, 3> rot;
  // quat[0] = gc_[3];
  // quat[1] = gc_[4];
  // quat[2] = gc_[5];
  // quat[3] = gc_[6];
  // raisim::quatToRotMat(quat, rot);
  //
  // cout<<rot.e().row(2)<<endl;

  //cout<<gc_[3]<<gc_[4]<<gc_[5]<<gc_[6]<<endl;

  // double qw1 = gc_[3];
  // Eigen::Vector3d qv1(gc_[4], gc_[5], gc_[6]);
  //
  // double theta;
  // if (qw1 > 0.99) {
  //     theta = acos(0.99) * 2;
  // }
  // else {
  //     theta = acos(qw1) * 2;
  // }
  //
  // double vv1, vv2, vv3;
  //
  // vv1 = qv1[0] / sin(theta/2);
  // vv2 = qv1[1] / sin(theta/2);
  // vv3 = qv1[2] / sin(theta/2);
  //
  // double v1, v2, v3;
  // v1 = -theta*vv1;
  // v2 = -theta*vv2;
  // v3 = -theta*vv3;
  //
  // cout<<v1<<endl;
  // cout<<v2<<endl;
  // cout<<v3<<endl;
  // gc_[3] = 0.6533; gc_[4] = 0.6533; gc_[5] = 0.2706; gc_[6] = 0.2706;
  //
  // double qw1 = gc_[3];
  // Eigen::Vector3d qv1(gc_[4], gc_[5], gc_[6]);
  //
  // double v1, v2, v3;
  //
  // // roll (x-axis rotation)
  // double sinr_cosp = 2 * (qw1 * qv1[0] + qv1[1] * qv1[2]);
  // double cosr_cosp = 1 - 2 * (qv1[0] * qv1[0] + qv1[1] * qv1[1]);
  // v1 = (std::atan2(sinr_cosp, cosr_cosp));
  //
  // // pitch (y-axis rotation)
  // double sinp = 2 * (qw1 * qv1[1] - qv1[2] * qv1[0]);
  // if (std::abs(sinp) >= 1)
  //     v2 = (std::copysign(M_PI / 2, sinp)); // use 90 degrees if out of range
  // else
  //     v2 = (std::asin(sinp));
  //
  // // yaw (z-axis rotation)
  // double siny_cosp = 2 * (qw1 * qv1[2] + qv1[0] * qv1[1]);
  // double cosy_cosp = 1 - 2 * (qv1[1] * qv1[1] + qv1[2] * qv1[2]);
  // v3 = (std::atan2(siny_cosp, cosy_cosp));
  //
  // cout<<v1<<endl;
  // cout<<v2<<endl;
  // cout<<v3<<endl;

  // Height map experiment end


  //robot->setGeneralizedCoordinate(gc_);




  // while (time_episode_ < 6){
  //
  //   //server.lockVisualizationServerMutex();
  //
  //   int n_walk_cycle = 0;
  //   double tau = 0;
  //   n_walk_cycle = (int) (time_episode_ / duration_ref_motion);
  //   tau = time_episode_ - n_walk_cycle * duration_ref_motion;
  //   get_ref_motion(ref_motion_one, tau);
  //   ref_motion_one[0] += (ref_motion_.back()[0] - ref_motion_[0][0]) * n_walk_cycle;
  //   ref_motion_one[2] += (ref_motion_.back()[2] - ref_motion_[0][2]) * n_walk_cycle + 0.03;
  //
  //   get_gv_init(gv_init_, tau);
  //
  //   //get_gait_motion(ref_motion_one, gait_motion_one);
  //   //robot->setState(ref_motion_one, gv_init_);
  //   ref_motion_one[1] += 40.33;
  //   robot->setGeneralizedCoordinate(ref_motion_one);
  //   //gv_init_ = 0.4 * gv_init_;
  //   // robot->setGeneralizedVelocity(gv_init_);
  //   //
  //   // for (int i = 0; i < 10; i++){
  //   // //robot->setGeneralizedVelocity(gv_init_);
  //   // world.integrate();
  //   // }
  //   // robot->setGeneralizedVelocity(gv_init_);
  //   // world.integrate();
  //
  //   //cout<<cumultime_.back()<<endl;
  //
  //   //if (t == 0){
  //   //}
  //   //vis->run();
  //
  //   // auto com_ = robot->getCompositeCOM();
  //   // cout<<com_<<endl;
  //
  //   // cout<<"gv_init_[27]: "<<gv_init_[27]<<endl;
  //   // cout<<"gv_init_[28]: "<<gv_init_[28]<<endl;
  //   // cout<<"gv_init_[29]: "<<gv_init_[29]<<endl;
  //
  //   time_episode_ = time_episode_ + dt;
  //
  //   //robot->getState(gc_, gv_);
  //
  //   //cout<<gc_[0]<<endl;
  //   //cout<<gc_[1]<<endl;
  //   //
  //   // map_height = ground->getHeight(gc_[0],gc_[1]);
  //   //
  //   // robot->getState(gc_, gv_);
  //   //
  //   // cout<<"gc_[2]: "<<gc_[2]<<endl;
  //   // cout<<"map_height: "<<map_height<<endl;
  //   //
  //   // for (int i = 0; i < 10; i++){
  //   // world.integrate();
  //   // }
  //
  //   // for (int i=0; i<10; i++) {
  //   //   std::this_thread::sleep_for(std::chrono::microseconds(1000));
  //   //   server.integrateWorldThreadSafe();
  //   // }
  //
  //   std::this_thread::sleep_for(std::chrono::microseconds(5000));
  //   //world.integrate();
  //   //server.unlockVisualizationServerMutex();
  //
  // }


  for (int i=0; i<200000000; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    //box->setVelocity(-1, 0, 0, 0, 0, 0);
    //server.integrateWorldThreadSafe();
  }

  server.killServer();

}

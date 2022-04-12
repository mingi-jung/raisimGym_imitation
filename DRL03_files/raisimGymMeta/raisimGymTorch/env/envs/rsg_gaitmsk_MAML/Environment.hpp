//
// Created by jemin on 3/27/19.
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

/* Convention
*
*   observation space = [ height                                                      n =  1, si =  0
*                         z-axis in world frame expressed in body frame (R_b.row(2))  n =  3, si =  1
*                         joint angles,                                               n = 28, si =  4
*                         body Linear velocities,                                     n =  3, si = 32
*                         body Angular velocities,                                    n =  3, si = 35
*                         joint velocities,                                           n = 28, si = 38 ] total 66
*/


#include <cstdlib>
#include <cstdint>
#include <set>
#include <raisim/RaisimServer.hpp>
#include "../../RaisimGymEnv.hpp"

#include <nlohmann/json.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include "DeepmimicUtility.hpp"
// #include "quaternion_mskbiodyn.hpp"

using namespace std;

namespace raisim {

    // class Quaternion
    // {
    // public:
    //     Quaternion(double wVal, double xVal, double yVal, double zVal) {
    //         w = wVal; x= xVal; y = yVal; z = zVal;
    //     }
    //
    //     Quaternion() {
    //         Quaternion(1, 0, 0, 0);
    //     }
    //
    //     double getW() { return w; }
    //     double getX() { return x; }
    //     double getY() { return y; }
    //     double getZ() { return z; }
    //     void set(double wVal, double xVal, double yVal, double zVal) {
    //         w = wVal; x= xVal; y = yVal; z = zVal;
    //     }
    //
    // private:
    //     double w, x, y, z;
    // };


    class ENVIRONMENT : public RaisimGymEnv {

    public:

      explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

            /// unity visualization server
            //server_.launchServer();

            /// Reward coefficients
            READ_YAML(double, angularPosRewardCoeff_, cfg["angularPosRewardCoeff"])
            READ_YAML(double, angularVelRewardCoeff_, cfg["angularVelRewardCoeff"])
            READ_YAML(double, endeffPosRewardCoeff_, cfg["endeffPosRewardCoeff"])
            READ_YAML(double, comPosRewardCoeff_, cfg["comPosRewardCoeff"])
            //READ_YAML(std::string, ref_motion_file_, cfg["ref_motion_file"])

            // mingi
            read_ref_motion1();
            read_ref_motion2();
            get_ref_motion_all();

            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_ + "/urdf/subject06_scaling.urdf");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            ground_ = world_->addHeightMap(resourceDir_ + "/KAIST_terrain/straight_80cm.png", 34, 1.05, 80, 80, 0.00027, -8.848); // subjedt06
            // ground_ = world_->addHeightMap(resourceDir_ + "/KAIST_terrain/straight_80cm.png", 34, -1.4, 80, 80, 0.00027, -8.848); // subjedt06
            // ground_ = world_->addHeightMap(resourceDir_ + "/KAIST_terrain/straight_80cm.png", 34, -0.2, 80, 80, 0.00027, -8.848); // subjedt06
            //ground_ = world_->addGround();

            ground_->setName("Ground");


            /// get robot data
            gcDim_ = anymal_->getGeneralizedCoordinateDim(); //43
            gvDim_ = anymal_->getGeneralizedVelocityDim(); //34
            nJoints_ = 25; // 34-6=28=1*4+3*8

            /// initialize containers
            gc_.setZero(gcDim_);
            gc_init_.setZero(gcDim_);
            gv_.setZero(gvDim_);
            gv_init_.setZero(gvDim_);
            torque_.setZero(gvDim_);
            pTarget_.setZero(gcDim_);
            vTarget_.setZero(gvDim_);
            pTarget28_.setZero(nJoints_);

            // mingi
            gc_ref_.setZero(gcDim_);
            gv_ref_.setZero(gvDim_);

            /// goal by mingi
            target_direction_ << 1, 0, 0;

            /// amp mingi
            obDouble_amp_prev_.setZero(71);
            obDouble_amp_present_.setZero(71);
            obDouble_amp_all_.setZero(142);
            ran_start_pos_ = 0;
            curri_num_1 = 1.02;
            curri_num_2 = 1.08;

            /// load urdf-specific joint information
            GetJointInfo();

            /// set pd gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            jointPgain.setZero();
            jointPgain.tail(nJoints_).setConstant(5.0);  // <----
            jointDgain.setZero();
            jointDgain.tail(nJoints_).setConstant(1.0);  // <----

            jointPgain <<
                    0, 0, 0,           // root translation
                    0, 0, 0,           // root rotation
                    //60, 40, 60,      // lumbar (RBend,  LRot, extention)
                    30, 20, 30,        // lumbar (RBend,  LRot, extention)
                    15, 15, 15,        // right shoulder (adduction, internal, flexion)
                    20,                // right elbow (flexion)
                    15, 15, 15,        // left shoulder (abduction, external, flexion)
                    20,                // left elbow (flexion)
                    30, 20, 60,       // right hip (adduction, internal, flexion)
                    60,               // right knee (extention)
                    10, 10, 20,        // right ankle (inversion, internal, dorsiflexion)
                    30, 20, 60,       // left hip (abduction, external, flexion)
                    60,               // left knee (extention)
                    10, 10, 20;        // left ankle (eversion, external, dorsiflexion)

            jointDgain <<
                    0, 0, 0,           // root translation
                    0, 0, 0,           // root rotation
                    10, 10, 10,      // lumbar (RBend,  LRot, extention)
                    4, 4, 4,        // right shoulder (adduction, internal, flexion)
                    3,                // right elbow (flexion)
                    4, 4, 4,        // left shoulder (abduction, external, flexion)
                    3,                // left elbow (flexion)
                    5, 5, 5,       // right hip (adduction, internal, flexion)
                    5,               // right knee (extention)
                    4, 4, 4,        // right ankle (inversion, internal, dorsiflexion)
                    5, 5, 5,       // left hip (abduction, external, flexion)
                    5,               // left knee (extention)
                    4, 4, 4;        // left ankle (eversion, external, dorsiflexion)

            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            actionDim_ = nJoints_;
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);

            obDim_ = 89; /// convention described on top
            obDim_amp_ = 71;
            obMean_.setZero(obDim_);
            obStd_.setZero(obDim_);

            /// action & observation scaling
            // actionMean_ = gc_init_.tail(nJoints_);
            // actionStd_.setConstant(0.6);

            actionMean_ <<
                    0, 0, 0,           // lumbar
                    0, 0, 0,           // right shoulder
                    1.5,               // right elbow
                    0, 0, 0,           // left shoulder
                    1.5,               // left elbow
                    0, 0, 0,           // right hip
                    0,              // right knee
                    0, 0, 0,           // right ankle
                    0, 0, 0,           // left hip
                    0,              // left knee
                    0, 0, 0;           // left ankle

            actionStd_ <<
                    1.061, 1.061, 1.061,     // lumbar  1.5/sqrt(2)
                    2.121, 2.121, 2.121,     // right shoulder   3.0/sqrt(2)
                    1.5,                     // right elbow
                    2.121, 2.121, 2.121,     // left shoulder    3.0/sqrt(2)
                    1.5,                     // left elbow
                    2.000, 1.000, 2.000,     // right hip        1.5/sqrt(2)
                    1.1,                     // right knee
                    1.061, 1.061, 1.061,     // right ankle      1.5/sqrt(2)
                    2.000, 1.000, 2.000,     // left hip         1.5/sqrt(2)
                    1.1,                     // left knee
                    1.061, 1.061, 1.061;     // left ankle       1.5/sqrt(2)

            obMean_ <<
                    0.91, /// average height
                    0.0, 0.0, 0.0, /// gravity axis 3
                    //gc_init_.tail(28),
                    0, 0, 0,           // lumbar
                    0, 0, 0,           // right shoulder
                    0,               // right elbow
                    0, 0, 0,           // left shoulder
                    0,               // left elbow
                    0, 0, 0,           // right hip
                    0,              // right knee
                    0, 0, 0,           // right ankle
                    0, 0, 0,           // left hip
                    0,              // left knee
                    0, 0, 0,           // left ankle
                    Eigen::VectorXd::Constant(6, 0.0), /// body lin/ang vel 6
                    Eigen::VectorXd::Constant(25, 0.0), /// joint vel history
                    Eigen::VectorXd::Constant(25, 0.0),
                    1.5, // target vel
                    1, 0, 0; // target direnction

            //  observation space = [ height                                                      n =  1, si =  0
            //                        z-axis in world frame expressed in body frame (R_b.row(2))  n =  3, si =  1
            //                        joint angles,                                               n = 28, si =  4
            //                        body Linear velocities,                                     n =  3, si = 32
            //                        body Angular velocities,                                    n =  3, si = 35
            //                        joint velocities,                                           n = 28, si = 38 ] total 66

            obStd_ <<
                    0.3, /// average height
                    Eigen::VectorXd::Constant(3, 0.5), /// gravity axes angles
                    1.061, 1.061, 1.061,     // lumbar  1.5/sqrt(2)
                    2.121, 2.121, 2.121,     // right shoulder   3.0/sqrt(2)
                    1.5,                     // right elbow
                    2.121, 2.121, 2.121,     // left shoulder    3.0/sqrt(2)
                    1.5,                     // left elbow
                    1.061, 1.061, 1.061,     // right hip        1.5/sqrt(2)
                    1.1,                     // right knee
                    1.061, 1.061, 1.061,     // right ankle      1.5/sqrt(2)
                    1.061, 1.061, 1.061,     // left hip         1.5/sqrt(2)
                    1.1,                     // left knee
                    1.061, 1.061, 1.061,     // left ankle       1.5/sqrt(2)
                    Eigen::VectorXd::Constant(3, 2.0), /// linear velocity
                    Eigen::VectorXd::Constant(3, 4.0), /// angular velocities
                    Eigen::VectorXd::Constant(25, 10.0), /// joint velocities
                    Eigen::VectorXd::Constant(25, 0.3),
                    0.1,
                    0, 0, 0;


            /// indices of links that should not make contact with ground
            footIndices_.insert(anymal_->getBodyIdx("toes_r"));
            footIndices_.insert(anymal_->getBodyIdx("calcn_r"));
            footIndices_.insert(anymal_->getBodyIdx("toes_l"));
            footIndices_.insert(anymal_->getBodyIdx("calcn_l"));



            /// visualize if it is the first environment
            //if (visualizable_) {
              server_ = std::make_unique<raisim::RaisimServer>(world_.get());
              server_->launchServer();
              server_->focusOn(anymal_);


              // visSphere1 = server_->addVisualSphere("v_sphere1", 0.2, 1, 0, 0, 1);
              // visSphere2 = server_->addVisualSphere("v_sphere2", 0.2, 1, 0, 0, 1);
              // visSphere3 = server_->addVisualSphere("v_sphere3", 0.2, 1, 0, 0, 1);
              // visSphere4 = server_->addVisualSphere("v_sphere4", 0.2, 1, 0, 0, 1);
              // visSphere5 = server_->addVisualSphere("v_sphere5", 0.2, 1, 0, 0, 1);
              // visSphere6 = server_->addVisualSphere("v_sphere6", 0.2, 1, 0, 0, 1);
              // visSphere7 = server_->addVisualSphere("v_sphere7", 0.2, 1, 0, 0, 1);
              // visSphere8 = server_->addVisualSphere("v_sphere8", 0.2, 1, 0, 0, 1);
              // visSphere9 = server_->addVisualSphere("v_sphere9", 0.2, 1, 0, 0, 1);
              // visSphere10 = server_->addVisualSphere("v_sphere10", 0.2, 1, 0, 0, 1);
              // visSphere11 = server_->addVisualSphere("v_sphere11", 0.2, 1, 0, 0, 1);
              // visSphere12 = server_->addVisualSphere("v_sphere12", 0.2, 1, 0, 0, 1);
              // visSphere13 = server_->addVisualSphere("v_sphere13", 0.2, 1, 0, 0, 1);
              // visSphere14 = server_->addVisualSphere("v_sphere14", 0.2, 1, 0, 0, 1);
              // visSphere15 = server_->addVisualSphere("v_sphere15", 0.2, 1, 0, 0, 1);
              // visSphere16 = server_->addVisualSphere("v_sphere16", 0.2, 1, 0, 0, 1);
              // visSphere17 = server_->addVisualSphere("v_sphere17", 0.2, 1, 0, 0, 1);
              // visSphere18 = server_->addVisualSphere("v_sphere18", 0.2, 1, 0, 0, 1);
              // visSphere19 = server_->addVisualSphere("v_sphere19", 0.2, 1, 0, 0, 1);
              // visSphere20 = server_->addVisualSphere("v_sphere20", 0.2, 1, 0, 0, 1);
              // visSphere21 = server_->addVisualSphere("v_sphere21", 0.2, 1, 0, 0, 1);
              // visSphere22 = server_->addVisualSphere("v_sphere22", 0.2, 1, 0, 0, 1);
              // visSphere23 = server_->addVisualSphere("v_sphere23", 0.2, 1, 0, 0, 1);
              // visSphere24 = server_->addVisualSphere("v_sphere24", 0.2, 1, 0, 0, 1);
              // visSphere25 = server_->addVisualSphere("v_sphere25", 0.2, 1, 0, 0, 1);

            //}
        }

        ~ENVIRONMENT() final = default;

        void init() final {}

        void reset() final {

            //cout<<"error check in reset() 1"<<endl;

            // double duration_ref_motion = cumultime_.back();

      			std::random_device rd;  // random device
      			std::mt19937 mersenne(rd()); // random generator, a mersenne twister
      			std::uniform_real_distribution<double> distribution(0.0, 1.0);
      			double random_number = distribution(mersenne);

            /// goal by mingi
            std::random_device rd2;  // random device
      			std::mt19937 mersenne2(rd2()); // random generator, a mersenne twister
      			// std::uniform_real_distribution<double> distribution2(0.65, curri_num_2);
            std::uniform_real_distribution<double> distribution2(curri_num_1, curri_num_2);
      			double random_number2 = distribution2(mersenne2);
            target_vel_ = random_number2;
            // target_vel_ = 0.8;
            // target_vel_ = 1.05;
            // cout<<"target_vel_: "<<target_vel_<<endl;

            // target_vel_ = 1.047;
            // target_vel_ = 1.328;
            // target_vel_ = 1.328;

      			// time_episode_ = random_number * duration_ref_motion;
            // time_episode_ = 0.01;

            Eigen::VectorXd ref_motion_one_start(37);
            ref_motion_one_start.setZero();


            // if (ran_start_pos_ < 3){
            //   double duration_ref_motion = cumultime1_.back();
            //   time_episode_ = random_number * duration_ref_motion;
            //   get_ref_motion1(ref_motion_one_start, time_episode_);
            //   get_gv_init1(gv_init_, time_episode_);
            //   ran_start_pos_ += 1;
            // }
            // else{
            //   double duration_ref_motion = cumultime2_.back();
            //   time_episode_ = random_number * duration_ref_motion;
            //   get_ref_motion2(ref_motion_one_start, time_episode_);
            //   get_gv_init2(gv_init_, time_episode_);
            //   ref_motion_one_start[1] += 1.05;
            //   // ref_motion_one_start[1] += 1.25;
            //   ran_start_pos_ = 0;
            // }

            double duration_ref_motion = cumultime1_.back();
            time_episode_ = random_number * duration_ref_motion;
            get_ref_motion1(ref_motion_one_start, time_episode_);
            get_gv_init1(gv_init_, time_episode_);

            // ref_motion_one_start[1] += 2.45; // slow walking start

            anymal_->setState(ref_motion_one_start, gv_init_);

            json_count_ = 0;

            /// amp mingi
            reset_check = 1;

            updateObservation();

            //cout<<"error check in reset() 2"<<endl;

        }

        float step(const Eigen::Ref<EigenVec> &action) final {
            //cout<<"error check in step() 1"<<endl;
            double v1[3], q1[4]; // temporary variables

            if (action.size() != actionDim_) {
                //cout << "action size mismatch!!!" << std::endl;
            }

            /// action scaling
            pTarget28_ = action.cast<double>();
            pTarget28_ = pTarget28_.cwiseProduct(actionStd_);
            pTarget28_ += actionMean_;

            // lumbar
            v1[0] = pTarget28_(0);
            v1[1] = pTarget28_(1);
            v1[2] = pTarget28_(2);
            vec2quat(q1, v1);
            pTarget_.segment(7, 4) << q1[0], q1[1], q1[2], q1[3];

            // right shoulder
            v1[0] = pTarget28_(3);
            v1[1] = pTarget28_(4);
            v1[2] = pTarget28_(5);
            vec2quat(q1, v1);
            pTarget_.segment(11, 4) << q1[0], q1[1], q1[2], q1[3];

            // right elbow
            pTarget_(15) = pTarget28_(6);

            // left shoulder
            v1[0] = pTarget28_(7);
            v1[1] = pTarget28_(8);
            v1[2] = pTarget28_(9);
            vec2quat(q1, v1);
            pTarget_.segment(16, 4) << q1[0], q1[1], q1[2], q1[3];

            // left elbow
            pTarget_(20) = pTarget28_(10);

            // right hip
            v1[0] = pTarget28_(11);
            v1[1] = pTarget28_(12);
            v1[2] = pTarget28_(13);
            vec2quat(q1, v1);
            pTarget_.segment(21, 4) << q1[0], q1[1], q1[2], q1[3];

            // right knee
            pTarget_(25) = pTarget28_(14);

            // right ankle
            pTarget_(26) = pTarget28_(15);
            pTarget_(27) = pTarget28_(16);
            pTarget_(28) = pTarget28_(17);

            // left hip
            v1[0] = pTarget28_(18);
            v1[1] = pTarget28_(19);
            v1[2] = pTarget28_(20);
            vec2quat(q1, v1);
            pTarget_.segment(29, 4) << q1[0], q1[1], q1[2], q1[3];

            // left knee
            pTarget_(33) = pTarget28_(21);

            // left ankle
            pTarget_(34) = pTarget28_(22);
            pTarget_(35) = pTarget28_(23);
            pTarget_(36) = pTarget28_(24);

            anymal_->setPdTarget(pTarget_, vTarget_);

            //cout<<"error check in step() 2"<<endl;

            // dynamics simulation and visualization
            auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);


            for (int i = 0; i < loopCount; i++) {

                if(server_) server_->lockVisualizationServerMutex();
                //cout<<"error check in step() 33"<<endl;

                world_->integrate();

                //cout<<"error check in step() 34"<<endl;

                if(server_) server_->unlockVisualizationServerMutex();
                //cout<<"error check in step() 35"<<endl;

                time_episode_ += (double) simulation_dt_;

            }

            //cout<<"error check in step() 4"<<endl;

            // update gc_ and gv_

            updateObservation();
            //cout<<"error check in step() 5"<<endl;

            // // perveption points
            // double qw10 = gc_[3];
            // Eigen::Vector3d qv10(gc_[4], gc_[5], gc_[6]);
            // double yaw0;
            //
            // double siny_cosp0 = 2 * (qw10 * qv10[2] + qv10[0] * qv10[1]);
            // double cosy_cosp0 = 1 - 2 * (qv10[1] * qv10[1] + qv10[2] * qv10[2]);
            // yaw0 = (std::atan2(siny_cosp0, cosy_cosp0));
            //
            //
            // int i = 0;
            // int j = 0;
            // i = 0; j = 0;
            // visSphere1->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 1; j = 0;
            // visSphere2->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 2; j = 0;
            // visSphere3->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 3; j = 0;
            // visSphere4->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 4; j = 0;
            // visSphere5->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 0; j = 1;
            // visSphere6->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 1; j = 1;
            // visSphere7->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 2; j = 1;
            // visSphere8->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 3; j = 1;
            // visSphere9->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 4; j = 1;
            // visSphere10->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 0; j = 2;
            // visSphere11->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 1; j = 2;
            // visSphere12->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 2; j = 2;
            // visSphere13->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 3; j = 2;
            // visSphere14->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 4; j = 2;
            // visSphere15->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 0; j = 3;
            // visSphere16->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 1; j = 3;
            // visSphere17->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 2; j = 3;
            // visSphere18->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 3; j = 3;
            // visSphere19->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 4; j = 3;
            // visSphere20->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 0; j = 4;
            // visSphere21->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 1; j = 4;
            // visSphere22->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 2; j = 4;
            // visSphere23->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 3; j = 4;
            // visSphere24->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            // i = 4; j = 4;
            // visSphere25->setPosition(gc_[0] + ((-0.6+0.3*i) * cos(yaw0) - (-0.6+0.3*j) * sin(yaw0)), gc_[1] + ((-0.6+0.3*i) * sin(yaw0) + (-0.6+0.3*j) * cos(yaw0)), ground_->getHeight(gc_[0], gc_[1]) + height_mat_(4-i,4-j) + 0.1);
            //
            //


            // time_episode_ += (double) control_dt_;

            /// goal by mingi

            // cout<<"bodyLinearVel_: "<<bodyLinearVel_[0]<<endl;

            // if (target_vel_ <4){
            //   target_vel_ += 0.0005;
            // }
            // // target_vel_ += 0.01;
            // cout<<target_vel_<<", "<<target_direction_.dot(bodyLinearVel_)<<endl;

            goalReward_ = exp(-0.25 * pow(target_vel_ - target_direction_.dot(bodyLinearVel_),2));

            totalReward_ = goalReward_;

            // get_angularPosReward(difference_angle_, gc_); // calculate the difference_angle_
            // angularPosReward_ = 0.3 * difference_angle_;
            //
            // totalReward_ = goalReward_ + angularPosReward_;

            // cout<<"goalReward_: "<<goalReward_<<endl;
            // cout<<"gtarget_direction_.dot(bodyLinearVel_): "<<target_direction_.dot(bodyLinearVel_)<<endl;
            // cout<<"angularPosReward_: "<<angularPosReward_<<endl;

            if (isnan(totalReward_) != 0){
      				is_nan_detected_ = true;

              goalReward_ = 0.;
      				totalReward_ = 0.;

      				//std:://cout<<"NaN detected"<<std::endl;
      			}

            return (float) totalReward_;

        }


        void updateExtraInfo() final {
            //cout<<"error check in ExtraInfo() 1"<<endl;
            //extraInfo_["forward vel reward"] = forwardVelReward_;
            //extraInfo_["base height"] = gc_[2];

            extraInfo_["total_reward"] = totalReward_;
            // extraInfo_["goalReward_"] = goalReward_;
            // extraInfo_["angularPosReward_"] = angularPosReward_;


            // auto a = anymal_->getGeneralizedCoordinate();
            // auto idx = anymal_->getMappingFromBodyIndexToGeneralizedCoordinateIndex()[anymal_->getBodyIdx("tibia_r")];
            // extraInfo_["Knee_Angle"] = -180*a[idx]/M_PI;
            //
            // auto a_ref = anymal_ref_->getGeneralizedCoordinate();
            // auto idx_ref = anymal_ref_->getMappingFromBodyIndexToGeneralizedCoordinateIndex()[anymal_->getBodyIdx("tibia_r")];
            // extraInfo_["Knee_Angle_Ref"] = -180*a_ref[idx_ref]/M_PI;


        }

        void curriculumUpdate() final {


          if (curri_num_1 > 0.8){
            curri_num_1 -= 0.0005;
            // curri_num_1 += 0.001;
          }

          if (curri_num_2 < 3.5){
            curri_num_2 += 0.0005;
            // curri_num_2 += 0.001;
          }

        }


        void updateObservation() {
            //cout<<"error check in Observation() 1"<<endl;
            double v1[3], q1[4]; // temporary variables

            anymal_->getState(gc_, gv_);
            //cout<<"error check in Observation() 2"<<endl;

            obDouble_.setZero(obDim_);
            obScaled_.setZero(obDim_);

            /// body height
            obDouble_[0] = gc_[2] - ground_->getHeight(gc_[0], gc_[1]);

            /// body orientation
            raisim::Vec<4> quat;
            raisim::Mat<3, 3> rot;
            quat[0] = gc_[3];
            quat[1] = gc_[4];
            quat[2] = gc_[5];
            quat[3] = gc_[6];
            raisim::quatToRotMat(quat, rot);
            obDouble_.segment(1, 3) = rot.e().row(2); // 20200510 SKOO which direction is it?

            // lumbar
            q1[0] = gc_(7);
            q1[1] = gc_(8);
            q1[2] = gc_(9);
            q1[3] = gc_(10);
            quat2vec(q1, v1);
            obDouble_.segment(4, 3) << v1[0], v1[1], v1[2];

            // right shoulder
            q1[0] = gc_(11);
            q1[1] = gc_(12);
            q1[2] = gc_(13);
            q1[3] = gc_(14);
            quat2vec(q1, v1);
            obDouble_.segment(7, 3) << v1[0], v1[1], v1[2];

            // right elbow
            obDouble_(10) = gc_(15);

            // left shoulder
            q1[0] = gc_(16);
            q1[1] = gc_(17);
            q1[2] = gc_(18);
            q1[3] = gc_(19);
            quat2vec(q1, v1);
            obDouble_.segment(11, 3) << v1[0], v1[1], v1[2];

            // left elbow
            obDouble_(14) = gc_(20);

            // right hip
            q1[0] = gc_(21);
            q1[1] = gc_(22);
            q1[2] = gc_(23);
            q1[3] = gc_(24);
            quat2vec(q1, v1);
            obDouble_.segment(15, 3) << v1[0], v1[1], v1[2];

            // right knee
            obDouble_(18) = gc_(25);

            // right ankle
            obDouble_(19) = gc_(26);
            obDouble_(20) = gc_(27);
            obDouble_(21) = gc_(28);

            // right hip
            q1[0] = gc_(29);
            q1[1] = gc_(30);
            q1[2] = gc_(31);
            q1[3] = gc_(32);
            quat2vec(q1, v1);
            obDouble_.segment(22, 3) << v1[0], v1[1], v1[2];

            // right knee
            obDouble_(25) = gc_(33);

            // right ankle
            obDouble_(26) = gc_(34);
            obDouble_(27) = gc_(35);
            obDouble_(28) = gc_(36);

            // obDouble_.segment(4, 36) = gc_.tail(36);
            //  observation space = [ height                                                      n =  1, si =  0
            //                        z-axis in world frame expressed in body frame (R_b.row(2))  n =  3, si =  1
            //                        joint angles,                                               n = 28, si =  4
            //                        body Linear velocities,                                     n =  3, si = 32
            //                        body Angular velocities,                                    n =  3, si = 35
            //                        joint velocities,                                           n = 28, si = 38 ] total 66

            /// body velocities
            //
            // treadmill_->getState(gc_treadmill_, gv_treadmill_);
            //
            // Eigen::Vector3d treadmill_vec;
            // treadmill_vec << gv_treadmill_[0], 0, 0;

            bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
            obDouble_(29) = bodyLinearVel_[0];
            obDouble_(30) = bodyLinearVel_[1];
            obDouble_(31) = bodyLinearVel_[2];

            //obDouble_.segment(29, 3) = bodyLinearVel_ - treadmill_vel_;
            obDouble_.segment(32, 3) = bodyAngularVel_;

            /// joint velocities
            obDouble_.segment(35, 25) = gv_.tail(25);

            height_mat_.setZero(5,5);
            get_height(gc_, height_mat_);

            for (int i=0; i<5; i++){
              for (int j=0; j<5; j++){
                obDouble_(60 + 5*i + j) = height_mat_(i,j);
              }
            }

            obDouble_(85) = target_vel_;
            obDouble_.segment(86, 3) << target_direction_[0], target_direction_[1], target_direction_[2];

            //cout<<"error check in Observation() 3"<<endl;

            //obScaled_ = (obDouble_ - obMean_).cwiseQuotient(obStd_);
        }


        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            //ob = obScaled_.cast<float>();
            //cout<<"error check in Observe() 1"<<endl;
            ob = obDouble_.cast<float>();
        }

        void observe_amp(Eigen::Ref<EigenVec> ob) final {

          Eigen::VectorXd obDouble_amp;
          obDouble_amp.setZero(obDim_amp_);
          anymal_->getState(gc_, gv_);

          int idx_begin = 0;

          raisim::Vec<4> quat;
          raisim::Mat<3, 3> rot;
          quat[0] = gc_[3];
          quat[1] = gc_[4];
          quat[2] = gc_[5];
          quat[3] = gc_[6];
          raisim::quatToRotMat(quat, rot);

          /// body velocities
          bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
          bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
          obDouble_amp.segment(idx_begin, 3) = bodyLinearVel_;
          idx_begin += 3;
          obDouble_amp.segment(idx_begin, 3) = bodyAngularVel_;
          idx_begin += 3;

          /// joint angles
          set_obDouble_from_gc(obDouble_amp, gc_, idx_begin);
          idx_begin += nJoints_;

          /// joint velocities
          obDouble_amp.segment(idx_begin, nJoints_) = gv_.tail(nJoints_);
          idx_begin += nJoints_;

          /// end_effector position
          std::size_t n_endeff = 5;
          Eigen::MatrixXd gc_endeff(n_endeff, 3);
          get_endeffPos(gc_endeff);

          obDouble_amp.segment(idx_begin, 3) = gc_endeff.row(0);
          idx_begin += 3;
          obDouble_amp.segment(idx_begin, 3) = gc_endeff.row(1);
          idx_begin += 3;
          obDouble_amp.segment(idx_begin, 3) = gc_endeff.row(2);
          idx_begin += 3;
          obDouble_amp.segment(idx_begin, 3) = gc_endeff.row(3);
          idx_begin += 3;
          obDouble_amp.segment(idx_begin, 3) = gc_endeff.row(4);
          idx_begin += 3;

          // cout<<"amp data: "<<obDouble_amp<<endl;

          ob = obDouble_amp.cast<float>();
        }


        bool isTerminalState(float &terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);

            //cout<<"error check in isTerminalState() 1"<<endl;

            /// 20200515 SKOO if a body other than feet has a contact with the ground

            for (auto &contact: anymal_->getContacts()) {

                if (contact.getPairObjectIndex() == ground_->getIndexInWorld()) {
                    if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {

                        time_episode_ = 0;

                        return true;
                    }
                }
            }

            if (is_nan_detected_ == true){
              is_nan_detected_ = false;

              time_episode_ = 0;

              return true;
            }

            if (gc_[0]*gc_[0] + gc_[1]*gc_[1] > 400){

              time_episode_ = 0;

              return true;

            }

            if (gc_[2] < 0){

              time_episode_ = 0;

              return true;

            }

            terminalReward = 0.f;
            return false;
        }


        void setSeed(int seed) final {
            //cout<<"error check in setSeed() 1"<<endl;
            std::srand(seed);
        }


        void close() final {
        }


        static void vec2quat(double *q1, double *v1) {
            double qrot; // rotation angle in radian
            double arot[3]; // rotation axis

            qrot = sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
            arot[0] = v1[0] / qrot;
            arot[1] = v1[1] / qrot;
            arot[2] = v1[2] / qrot;

            if (qrot < 0.001) {
                q1[0] = 1;
                q1[1] = 0;
                q1[2] = 0;
                q1[3] = 0;
            } else {
                q1[0] = cos(qrot / 2.0);
                q1[1] = arot[0] * sin(qrot / 2.0);
                q1[2] = arot[1] * sin(qrot / 2.0);
                q1[3] = arot[2] * sin(qrot / 2.0);
            }

        }


        static void quat2vec(double *q1, double *v1) {
            double qrot; // rotation angle in radian
            double arot[3]; // rotation axis

            qrot = acos(q1[0]) * 2;

            if (qrot < 0.001) {
                v1[0] = 0;
                v1[1] = 0;
                v1[2] = 0;
            } else {
                arot[0] = q1[1] / sin(qrot / 2);
                arot[1] = q1[2] / sin(qrot / 2);
                arot[2] = q1[3] / sin(qrot / 2);

                v1[0] = arot[0] * qrot;
                v1[1] = arot[1] * qrot;
                v1[2] = arot[2] * qrot;
            }

        }


        static void slerp(Quaternion &qout, Quaternion q1, Quaternion q2, double lambda)
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

        static raisim::Vec<3> pt_coord_txf(raisim::Mat<3,3> &rot, raisim::Vec<3> &trs, raisim::Vec<3> &pt) {
            raisim::Vec<3> pt_txfed = {
                    rot[0] * pt[0] + rot[3] * pt[1] + rot[6] * pt[2] + trs[0],
                    rot[1] * pt[0] + rot[4] * pt[1] + rot[7] * pt[2] + trs[1],
                    rot[2] * pt[0] + rot[5] * pt[1] + rot[8] * pt[2] + trs[2]
            };

            return pt_txfed;
        }


        template<typename Derived>
        inline bool is_finite(const Eigen::MatrixBase<Derived> &x) {
            return ((x - x).array() == (x - x).array()).all();
        }

        template<typename Derived>
        inline bool is_nan(const Eigen::MatrixBase<Derived> &x) {
            return ((x.array() == x.array())).all();
        }

        // prototypes of public functions
        void get_ref_motion1(Eigen::VectorXd &, double);
        void get_ref_motion2(Eigen::VectorXd &, double);
        void read_ref_motion1();
        void read_ref_motion2();
        void calc_slerp_joint(Eigen::VectorXd &, const int *, std::vector<Eigen::VectorXd> &, int, int, double);
        void calc_interp_joint(Eigen::VectorXd &, int , std::vector<Eigen::VectorXd> &, int, int, double);
        void get_gv_init1(Eigen::VectorXd &, double);
        void get_gv_init2(Eigen::VectorXd &, double);
        void get_height(Eigen::VectorXd &, Eigen::MatrixXd &);

        void set_obDouble_from_gc(Eigen::VectorXd &obDouble, Eigen::VectorXd &gc, int idx_begin);
        void get_endeffPos(Eigen::MatrixXd &gc_endeff);
        void GetJointInfo();
        void get_ref_motion_all();
        void get_angularPosReward(double &, Eigen::VectorXd &);


    private:
        int gcDim_, gvDim_, nJoints_, gcDim_treadmill_, gvDim_treadmill_;
        bool visualizable_ = false;
        std::normal_distribution<double> distribution_;
        raisim::ArticulatedSystem *anymal_, *anymal_ref_, *treadmill_;
        //raisim::Ground *ground_;
        raisim::HeightMap *ground_;
        //std::vector<GraphicObject> *anymalVisual_;
        //std::vector<GraphicObject> *anymalVisual_ref;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget28_, vTarget_, torque_, gc_treadmill_, gv_treadmill_, gc_init_treadmill_, gv_init_treadmill_;
        Eigen::VectorXd gc_ref_, gv_ref_;
        double desired_fps_ = 60.;
        double terminalRewardCoeff_ = 0; //-10.;

        double angularPosRewardCoeff_ = 0.; // First deepmimic reward
        double angularVelRewardCoeff_ = 0.; // Second deepmimic reward
        double endeffPosRewardCoeff_ = 0.; // Third deepmimic reward
        double comPosRewardCoeff_ = 0.; // Fourth deepmimic reward

        double angularPosReward_ = 0.; // First deepmimic reward
        double angularVelReward_ = 0.; // Second deepmimic reward
        double endeffPosReward_ = 0.; // Third deepmimic reward
        double comPosReward_ = 0.; // Fourth deepmimic reward
        double totalReward_ = 0.;

        Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
        Eigen::VectorXd obDouble_, obScaled_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        std::set<size_t> footIndices_;
        //raisim::RaisimServer server_;  // 20200610 unity visualization server

        // mingi
        std::vector<Eigen::VectorXd> ref_motion1_;
        std::vector<Eigen::VectorXd> ref_motion2_;
        std::string ref_motion_file_;
        std::vector<double> cumultime1_;
        std::vector<double> cumultime2_;
        double time_ref_motion_start_ = 0.;
        double time_episode_ = 0.;

        double difference_angle_;
        double difference_velocity_;
        double difference_end_;
        double difference_com_;

        Eigen::MatrixXd height_mat_;

        raisim::World world_ref_;

        bool is_nan_detected_ = false;

        nlohmann::json outputs;
        raisim::Vec<3> ankle_position;
        int json_count_ = 0;

        int treadmill_control_count = 0;

        raisim::Visuals *visSphere1;
        raisim::Visuals *visSphere2;
        raisim::Visuals *visSphere3;
        raisim::Visuals *visSphere4;
        raisim::Visuals *visSphere5;
        raisim::Visuals *visSphere6;
        raisim::Visuals *visSphere7;
        raisim::Visuals *visSphere8;
        raisim::Visuals *visSphere9;
        raisim::Visuals *visSphere10;
        raisim::Visuals *visSphere11;
        raisim::Visuals *visSphere12;
        raisim::Visuals *visSphere13;
        raisim::Visuals *visSphere14;
        raisim::Visuals *visSphere15;
        raisim::Visuals *visSphere16;
        raisim::Visuals *visSphere17;
        raisim::Visuals *visSphere18;
        raisim::Visuals *visSphere19;
        raisim::Visuals *visSphere20;
        raisim::Visuals *visSphere21;
        raisim::Visuals *visSphere22;
        raisim::Visuals *visSphere23;
        raisim::Visuals *visSphere24;
        raisim::Visuals *visSphere25;

        //raisim::Vec<3> end_pos0, end_pos1, end_pos2, end_pos3, end_pos4, end_pos5, end_pos6, end_pos7;


        /// AMP
        Eigen::VectorXd obDouble_amp_prev_;
        Eigen::VectorXd obDouble_amp_present_;
        Eigen::VectorXd obDouble_amp_all_;
        int reset_check = 1;
        std::vector<JointInfo> joint_info_;

        /// goal by mingi
        double target_vel_;
        Eigen::Vector3d target_direction_;
        double goalReward_;

        std::vector<Eigen::VectorXd> ref_motions_all_;
        int ran_start_pos_;
        // double difference_angle_;
        // double angularPosReward_ = 0.;

        double curri_num_1;
        double curri_num_2;

    };


    void ENVIRONMENT::read_ref_motion1() {
        Eigen::VectorXd jointNominalConfig(37);
        double totaltime = 0;
        //std::ifstream infile1(ref_motion_file_);
        std::ifstream infile1("/home/mingi/raisim_v6_workspace/raisimLib/raisimGymMeta/raisimGymTorch/env/envs/rsg_gaitmsk_MAML/rsc/refmotions/subject06_walking13_straight03_cyclic.txt");
        nlohmann::json  jsondata;
        infile1 >> jsondata;

        cumultime1_.push_back(0);

        Eigen::VectorXd jointIdxConvTable(37);
        jointIdxConvTable <<
            0, 2, 1,           // root translation
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
            double qw3;
            Eigen::Vector3d qv3(0, 0, 0);

            qw3 = qw2 * qw1 - qv2.dot(qv1);
            qv3 = qw2 * qv1 + qw1 * qv2 + qv2.cross(qv1);

            jointNominalConfig[3] = qw3;
            jointNominalConfig[4] = qv3[0];
            jointNominalConfig[5] = qv3[1];
            jointNominalConfig[6] = qv3[2];
            // x direction 90 degree rotation_end

            ref_motion1_.push_back(jointNominalConfig);

            totaltime += (double) jsondata["Frames"][iframe][0];
            cumultime1_.push_back(totaltime);
        }
        cumultime1_.pop_back();
    }

    void ENVIRONMENT::read_ref_motion2() {
        Eigen::VectorXd jointNominalConfig(37);
        double totaltime = 0;
        //std::ifstream infile1(ref_motion_file_);
        std::ifstream infile1("/home/opensim2020/raisim_v2_workspace/raisimlib/raisimGymTorchBio/raisimGymTorch/env/envs/rsg_mingi/rsc/motions/mingi_run_v2.txt");
        nlohmann::json  jsondata;
        infile1 >> jsondata;

        cumultime2_.push_back(0);

        Eigen::VectorXd jointIdxConvTable(37);
        jointIdxConvTable <<
            0, 2, 1,           // root translation
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
            double qw3;
            Eigen::Vector3d qv3(0, 0, 0);

            qw3 = qw2 * qw1 - qv2.dot(qv1);
            qv3 = qw2 * qv1 + qw1 * qv2 + qv2.cross(qv1);

            jointNominalConfig[3] = qw3;
            jointNominalConfig[4] = qv3[0];
            jointNominalConfig[5] = qv3[1];
            jointNominalConfig[6] = qv3[2];
            // x direction 90 degree rotation_end

            ref_motion2_.push_back(jointNominalConfig);

            totaltime += (double) jsondata["Frames"][iframe][0];
            cumultime2_.push_back(totaltime);
        }
        cumultime2_.pop_back();
    }

    void ENVIRONMENT::get_ref_motion_all() {

        double duration_ref_motion1 = cumultime1_.back();
        double duration_ref_motion2 = cumultime2_.back();

        double tau = 0;
        while (tau < duration_ref_motion1) {

          Eigen::VectorXd ref_motion_ins(37);
          ref_motion_ins.setZero();

          int idx_frame_prev = -1;
          int idx_frame_next = -1;
          double t_offset;
          double t_offset_ratio;

          for (int i = 1; i < cumultime1_.size(); i++) { // mingi 20200706 checked
              if (tau < cumultime1_[i]) {
                  idx_frame_prev = i - 1; // this index is including 0
                  idx_frame_next = i;
                  break;
              }
          }
          t_offset = tau - cumultime1_[idx_frame_prev];
          t_offset_ratio = t_offset / (cumultime1_[idx_frame_next] - cumultime1_[idx_frame_prev]);

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

          calc_slerp_joint(ref_motion_ins, idx_qroot, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_slerp_joint(ref_motion_ins, idx_qlumbar, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_slerp_joint(ref_motion_ins, idx_qrshoulder, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_slerp_joint(ref_motion_ins, idx_qlshoulder, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_slerp_joint(ref_motion_ins, idx_qrhip, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_slerp_joint(ref_motion_ins, idx_qlhip, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);

          calc_interp_joint(ref_motion_ins, idx_rootx, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_rooty, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_rootz, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_relbow, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_lelbow, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_rknee, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_rankle, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_rsubtalar, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_rmtp, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_lknee, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_lankle, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_lsubtalar, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
          calc_interp_joint(ref_motion_ins, idx_lmtp, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);


          ref_motions_all_.push_back(ref_motion_ins);
          tau += 0.01;
        }

        tau = 0;

        while (tau < duration_ref_motion2) {

            Eigen::VectorXd ref_motion_ins(37);
            ref_motion_ins.setZero();

            int idx_frame_prev = -1;
            int idx_frame_next = -1;
            double t_offset;
            double t_offset_ratio;

            for (int i = 1; i < cumultime2_.size(); i++) { // mingi 20200706 checked
                if (tau < cumultime2_[i]) {
                    idx_frame_prev = i - 1; // this index is including 0
                    idx_frame_next = i;
                    break;
                }
            }

            t_offset = tau - cumultime2_[idx_frame_prev];
            t_offset_ratio = t_offset / (cumultime2_[idx_frame_next] - cumultime2_[idx_frame_prev]);

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

            calc_slerp_joint(ref_motion_ins, idx_qroot, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_slerp_joint(ref_motion_ins, idx_qlumbar, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_slerp_joint(ref_motion_ins, idx_qrshoulder, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_slerp_joint(ref_motion_ins, idx_qlshoulder, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_slerp_joint(ref_motion_ins, idx_qrhip, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_slerp_joint(ref_motion_ins, idx_qlhip, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);

            calc_interp_joint(ref_motion_ins, idx_rootx, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_rooty, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_rootz, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_relbow, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_lelbow, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_rknee, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_rankle, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_rsubtalar, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_rmtp, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_lknee, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_lankle, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_lsubtalar, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
            calc_interp_joint(ref_motion_ins, idx_lmtp, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);

            ref_motions_all_.push_back(ref_motion_ins);

            tau += 0.01;
          }
    }


    void ENVIRONMENT::get_ref_motion1(Eigen::VectorXd &ref_motion_one, double tau) {
        int idx_frame_prev = -1;
        int idx_frame_next = -1;
        double t_offset;
        double t_offset_ratio;

        // SKOO 20200629 This loop should be checked again.
        for (int i = 1; i < cumultime1_.size(); i++) { // mingi 20200706 checked
            if (tau < cumultime1_[i]) {
                idx_frame_prev = i - 1; // this index is including 0
                idx_frame_next = i;
                break;
            }
        }

        t_offset = tau - cumultime1_[idx_frame_prev];
        t_offset_ratio = t_offset / (cumultime1_[idx_frame_next] - cumultime1_[idx_frame_prev]);

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

        calc_slerp_joint(ref_motion_one, idx_qroot, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qlumbar, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qrshoulder, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qlshoulder, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qrhip, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qlhip, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);

        calc_interp_joint(ref_motion_one, idx_rootx, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rooty, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rootz, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_relbow, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lelbow, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rknee, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rankle, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rsubtalar, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rmtp, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lknee, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lankle, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lsubtalar, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lmtp, ref_motion1_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    }

    void ENVIRONMENT::get_ref_motion2(Eigen::VectorXd &ref_motion_one, double tau) {
        int idx_frame_prev = -1;
        int idx_frame_next = -1;
        double t_offset;
        double t_offset_ratio;

        // SKOO 20200629 This loop should be checked again.
        for (int i = 1; i < cumultime2_.size(); i++) { // mingi 20200706 checked
            if (tau < cumultime2_[i]) {
                idx_frame_prev = i - 1; // this index is including 0
                idx_frame_next = i;
                break;
            }
        }

        t_offset = tau - cumultime2_[idx_frame_prev];
        t_offset_ratio = t_offset / (cumultime2_[idx_frame_next] - cumultime2_[idx_frame_prev]);

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

        calc_slerp_joint(ref_motion_one, idx_qroot, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qlumbar, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qrshoulder, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qlshoulder, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qrhip, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_slerp_joint(ref_motion_one, idx_qlhip, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);

        calc_interp_joint(ref_motion_one, idx_rootx, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rooty, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rootz, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_relbow, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lelbow, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rknee, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rankle, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rsubtalar, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_rmtp, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lknee, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lankle, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lsubtalar, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
        calc_interp_joint(ref_motion_one, idx_lmtp, ref_motion2_, idx_frame_prev, idx_frame_next, t_offset_ratio);
    }

    void ENVIRONMENT::calc_slerp_joint(
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

        Quaternion qprev(ref_motion[idxprev][idxW], ref_motion[idxprev][idxX],
                         ref_motion[idxprev][idxY], ref_motion[idxprev][idxZ]);
        Quaternion qnext(ref_motion[idxnext][idxW], ref_motion[idxnext][idxX],
                         ref_motion[idxnext][idxY], ref_motion[idxnext][idxZ]);
        Quaternion qout;
        slerp(qout, qprev, qnext, t_offset_ratio);
        ref_motion_one[idxW] = qout.getW();
        ref_motion_one[idxX] = qout.getX();
        ref_motion_one[idxY] = qout.getY();
        ref_motion_one[idxZ] = qout.getZ();
    }


    void ENVIRONMENT::calc_interp_joint(
            Eigen::VectorXd &ref_motion_one,                // output
            int idxjoint,                                   // index of a scalar joint to be updated
            std::vector<Eigen::VectorXd> &ref_motion,       // input
            int idxprev,                                    // input
            int idxnext,                                    // input
            double t_offset_ratio) {                        // input

        double a = ref_motion[idxprev][idxjoint];
        double b = ref_motion[idxnext][idxjoint];

        ref_motion_one[idxjoint] = a + (b - a) * t_offset_ratio;
    }

    void ENVIRONMENT::get_angularPosReward(
            double &difference_angle_,                       // output
            Eigen::VectorXd &gc_){

        difference_angle_ = 10000;

        for (int i = 0; i < ref_motions_all_.size(); i++){

          double difference_quat = 0;
          double difference_rev = 0;
          double difference_instant = 0;

          for (int j = 0; j < 37; j++) { // all index of motion
              if (j == 7 || j == 11 || j == 16 || j == 21 || j == 29) {
                  double qw1 = ref_motions_all_[i][j];
                  Eigen::Vector3d qv1(ref_motions_all_[i][j + 1], ref_motions_all_[i][j + 2], ref_motions_all_[i][j + 3]);
                  double qw2 = gc_[j];
                  Eigen::Vector3d qv2(gc_[j + 1], gc_[j + 2], gc_[j + 3]);
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

                  if (qw4 > 0.99) {
                      qw4 = acos(0.99) * 2;
                  }
                  else {
                      qw4 = acos(qw4) * 2;
                  }

                  difference_quat = difference_quat + qw4 * qw4;
              }

              if (j == 15 || j == 20 || j == 25 || j == 26 || j == 27 || j == 28 || j == 33 || j == 34 || j == 35 || j == 36) {

                  difference_rev = difference_rev + ((ref_motions_all_[i][j] - gc_[j]) * (ref_motions_all_[i][j] - gc_[j]));
              }

          }
          difference_instant = difference_quat + difference_rev;
          if (difference_instant < difference_angle_){
            difference_angle_ = difference_instant;
          }
        }
        // cout<<difference_angle_<<endl;
        difference_angle_ = exp(-0.04 * difference_angle_);

    }


    void ENVIRONMENT::get_gv_init1(
            Eigen::VectorXd &gv_init_, //output
            double tau){

        double duration_ref_motion = cumultime1_.back(); // in case, tau - control_dt is minus

        Eigen::VectorXd ref_motion_one_next(37);
        Eigen::VectorXd ref_motion_one_prev(37); //mingi, to calculate velocity by 미분


        double small_dt = 0.004;

        if (tau + small_dt/2 <= duration_ref_motion){
            get_ref_motion1(ref_motion_one_next, tau + small_dt/2);
        }
        else{ // if tau - control_dt is minus
            get_ref_motion1(ref_motion_one_next, tau + small_dt/2 - duration_ref_motion);
        }

        if (tau - small_dt/2 >= 0){
            get_ref_motion1(ref_motion_one_prev, tau - small_dt/2);
        }
        else{ // if tau - control_dt is minus
            get_ref_motion1(ref_motion_one_prev, duration_ref_motion + tau - small_dt/2);
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
    }

    void ENVIRONMENT::get_gv_init2(
            Eigen::VectorXd &gv_init_, //output
            double tau){

        double duration_ref_motion = cumultime2_.back(); // in case, tau - control_dt is minus

        Eigen::VectorXd ref_motion_one_next(37);
        Eigen::VectorXd ref_motion_one_prev(37); //mingi, to calculate velocity by 미분


        double small_dt = 0.004;

        if (tau + small_dt/2 <= duration_ref_motion){
            get_ref_motion2(ref_motion_one_next, tau + small_dt/2);
        }
        else{ // if tau - control_dt is minus
            get_ref_motion2(ref_motion_one_next, tau + small_dt/2 - duration_ref_motion);
        }

        if (tau - small_dt/2 >= 0){
            get_ref_motion2(ref_motion_one_prev, tau - small_dt/2);
        }
        else{ // if tau - control_dt is minus
            get_ref_motion2(ref_motion_one_prev, duration_ref_motion + tau - small_dt/2);
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
    }

    void ENVIRONMENT::get_height(Eigen::VectorXd &gc_, Eigen::MatrixXd &height_mat_) {

      double qw1 = gc_[3];
      Eigen::Vector3d qv1(gc_[4], gc_[5], gc_[6]);
      double yaw;

      double siny_cosp = 2 * (qw1 * qv1[2] + qv1[0] * qv1[1]);
      double cosy_cosp = 1 - 2 * (qv1[1] * qv1[1] + qv1[2] * qv1[2]);
      yaw = (std::atan2(siny_cosp, cosy_cosp));

      // for (int i=0; i<5; i++){
      //   for (int j=0; j<5; j++){
      //     height_mat_(4-i,4-j) = ground_->getHeight(gc_[0] + ((-0.6+0.3*i) * cos(yaw) - (-0.6+0.3*j) * sin(yaw)), gc_[1] + ((-0.6+0.3*i) * sin(yaw) + (-0.6+0.3*j) * cos(yaw))) - ground_->getHeight(gc_[0], gc_[1]);
      //   }
      // }

      for (int i=0; i<5; i++){
        for (int j=0; j<5; j++){
          height_mat_(4-i,4-j) = ground_->getHeight(gc_[0] + ((-0.6+0.3*i) * cos(yaw) - (-0.6+0.3*j) * sin(yaw)), gc_[1] + ((-0.6+0.3*i) * sin(yaw) + (-0.6+0.3*j) * cos(yaw))) - ground_->getHeight(gc_[0], gc_[1]);
          if (height_mat_(4-i,4-j) < -1){
            height_mat_(4-i,4-j) = -1;
          }
        }
      }

    }

    void ENVIRONMENT::GetJointInfo() {
        JointInfo joint_root_trs;       joint_info_.push_back(joint_root_trs);
        JointInfo joint_root_rot;       joint_info_.push_back(joint_root_rot);
        JointInfo joint_lumbar;         joint_info_.push_back(joint_lumbar);
        JointInfo joint_right_shoulder; joint_info_.push_back(joint_right_shoulder);
        JointInfo joint_right_elbow;    joint_info_.push_back(joint_right_elbow);
        JointInfo joint_left_shoulder;  joint_info_.push_back(joint_left_shoulder);
        JointInfo joint_left_elbow;     joint_info_.push_back(joint_left_elbow);
        JointInfo joint_right_hip;      joint_info_.push_back(joint_right_hip);
        JointInfo joint_right_knee;     joint_info_.push_back(joint_right_knee);
        JointInfo joint_right_ankle;    joint_info_.push_back(joint_right_ankle);
        JointInfo joint_right_subtalar; joint_info_.push_back(joint_right_subtalar);
        JointInfo joint_right_mtp;      joint_info_.push_back(joint_right_mtp);
        JointInfo joint_left_hip;       joint_info_.push_back(joint_left_hip);
        JointInfo joint_left_knee;      joint_info_.push_back(joint_left_knee);
        JointInfo joint_left_ankle;     joint_info_.push_back(joint_left_ankle);
        JointInfo joint_left_subtalar;  joint_info_.push_back(joint_left_subtalar);
        JointInfo joint_left_mtp;       joint_info_.push_back(joint_left_mtp);

        joint_info_[ 0].set_name_type_gc_gv("root_trs",   "trs3", 3, 3);
        joint_info_[ 1].set_name_type_gc_gv("root_rot",   "rot3", 4, 3);
        joint_info_[ 2].set_name_type_gc_gv("lumbar",     "rot3", 4, 3);
        joint_info_[ 3].set_name_type_gc_gv("r_shoulder", "rot3", 4, 3);
        joint_info_[ 4].set_name_type_gc_gv("r_elbow",    "rot1", 1, 1);
        joint_info_[ 5].set_name_type_gc_gv("l_shoulder", "rot3", 4, 3);
        joint_info_[ 6].set_name_type_gc_gv("l_elbow",    "rot1", 1, 1);
        joint_info_[ 7].set_name_type_gc_gv("r_hip",      "rot3", 4, 3);
        joint_info_[ 8].set_name_type_gc_gv("r_knee",     "rot1", 1, 1);
        joint_info_[ 9].set_name_type_gc_gv("r_ankle",    "rot1", 1, 1);
        joint_info_[10].set_name_type_gc_gv("r_subtalar", "rot1", 1, 1);
        joint_info_[11].set_name_type_gc_gv("r_mtp",      "rot1", 1, 1);
        joint_info_[12].set_name_type_gc_gv("l_hip",      "rot3", 4, 3);
        joint_info_[13].set_name_type_gc_gv("l_knee",     "rot1", 1, 1);
        joint_info_[14].set_name_type_gc_gv("l_ankle",    "rot1", 1, 1);
        joint_info_[15].set_name_type_gc_gv("l_subtalar", "rot1", 1, 1);
        joint_info_[16].set_name_type_gc_gv("l_mtp",      "rot1", 1, 1);

        joint_info_[ 0].set_gcidx(0, 1, 2, -1);    // root_trs;
        joint_info_[ 1].set_gcidx(3, 4, 5, 6);     // root_rot;
        joint_info_[ 2].set_gcidx(7, 8, 9, 10);    // lumbar;
        joint_info_[ 3].set_gcidx(11, 12, 13, 14); // right_shoulder;
        joint_info_[ 4].set_gcidx(15, -1, -1, -1); // right_elbow;
        joint_info_[ 5].set_gcidx(16, 17, 18, 19); // left_shoulder;
        joint_info_[ 6].set_gcidx(20, -1, -1, -1); // left_elbow;
        joint_info_[ 7].set_gcidx(21, 22, 23, 24); // right_hip;
        joint_info_[ 8].set_gcidx(25, -1, -1, -1); // right_knee;
        joint_info_[ 9].set_gcidx(26, -1, -1, -1); // right_ankle;
        joint_info_[10].set_gcidx(27, -1, -1, -1); // right_subtalar;
        joint_info_[11].set_gcidx(28, -1, -1, -1); // right_mtp;
        joint_info_[12].set_gcidx(29, 30, 31, 32); // left_hip;
        joint_info_[13].set_gcidx(33, -1, -1, -1); // left_knee;
        joint_info_[14].set_gcidx(34, -1, -1, -1); // left_ankle;
        joint_info_[15].set_gcidx(35, -1, -1, -1); // left_subtalar;
        joint_info_[16].set_gcidx(36, -1, -1, -1); // left_mtp;

        joint_info_[ 0].set_gvidx(0, 1, 2);        // root_trs;
        joint_info_[ 1].set_gvidx(3, 4, 5);        // root_rot;
        joint_info_[ 2].set_gvidx(6, 7, 8);        // lumbar;
        joint_info_[ 3].set_gvidx(9, 10, 11);      // right_shoulder;
        joint_info_[ 4].set_gvidx(12, -1, -1);     // right_elbow;
        joint_info_[ 5].set_gvidx(13, 14, 15);     // left_shoulder;
        joint_info_[ 6].set_gvidx(16, -1, -1);     // left_elbow;
        joint_info_[ 7].set_gvidx(17, 18, 19);     // right_hip;
        joint_info_[ 8].set_gvidx(20, -1, -1);     // right_knee;
        joint_info_[ 9].set_gvidx(21, -1, -1);     // right_ankle;
        joint_info_[10].set_gvidx(22, -1, -1);     // right_subtalar;
        joint_info_[11].set_gvidx(23, -1, -1);     // right_mtp;
        joint_info_[12].set_gvidx(24, 25, 26);     // left_hip;
        joint_info_[13].set_gvidx(27, -1, -1);     // left_knee;
        joint_info_[14].set_gvidx(28, -1, -1);     // left_ankle;
        joint_info_[15].set_gvidx(29, -1, -1);     // left_subtalar;
        joint_info_[16].set_gvidx(30, -1, -1);     // left_mtp;

        // actions for PD controller
        joint_info_[ 0].set_pos_actionmean(0.0, 0.0, 0.0);     // root_trs;
        joint_info_[ 1].set_pos_actionmean(0.0, 0.0, 0.0);     // root_rot;
        joint_info_[ 2].set_pos_actionmean(0.0, 0.0, 0.0);     // lumbar;
        joint_info_[ 3].set_pos_actionmean(0.0, 0.0, 0.0);     // right_shoulder;
        joint_info_[ 4].set_pos_actionmean(1.5, 0.0, 0.0);     // right_elbow;
        joint_info_[ 5].set_pos_actionmean(0.0, 0.0, 0.0);     // left_shoulder;
        joint_info_[ 6].set_pos_actionmean(1.5, 0.0, 0.0);     // left_elbow;
        joint_info_[ 7].set_pos_actionmean(0.0, 0.0, 0.0);     // right_hip;
        joint_info_[ 8].set_pos_actionmean(0.0, 0.0, 0.0);     // right_knee;
        joint_info_[ 9].set_pos_actionmean(0.0, 0.0, 0.0);     // right_ankle;
        joint_info_[10].set_pos_actionmean(0.0, 0.0, 0.0);     // right_subtalar;
        joint_info_[11].set_pos_actionmean(0.0, 0.0, 0.0);     // right_mtp;
        joint_info_[12].set_pos_actionmean(0.0, 0.0, 0.0);     // left_hip;
        joint_info_[13].set_pos_actionmean(0.0, 0.0, 0.0);     // left_knee;
        joint_info_[14].set_pos_actionmean(0.0, 0.0, 0.0);     // left_ankle;
        joint_info_[15].set_pos_actionmean(0.0, 0.0, 0.0);     // left_subtalar;
        joint_info_[16].set_pos_actionmean(0.0, 0.0, 0.0);     // left_mtp;

        joint_info_[ 0].set_pos_actionstd(0.0, 0.0, 0.0);        // root_trs;
        joint_info_[ 1].set_pos_actionstd(0.0, 0.0, 0.0);        // root_rot;
        joint_info_[ 2].set_pos_actionstd(1.061, 1.061, 1.061);  // lumbar;
        joint_info_[ 3].set_pos_actionstd(2.121, 2.121, 2.121);  // right_shoulder;
        joint_info_[ 4].set_pos_actionstd(1.5, 0.0, 0.0);        // right_elbow;
        joint_info_[ 5].set_pos_actionstd(2.121, 2.121, 2.121);  // left_shoulder;
        joint_info_[ 6].set_pos_actionstd(1.5, 0.0, 0.0);        // left_elbow;
        joint_info_[ 7].set_pos_actionstd(2.0, 1.0, 1.0);        // right_hip;
        joint_info_[ 8].set_pos_actionstd(1.1, 0.0, 0.0);        // right_knee;
        joint_info_[ 9].set_pos_actionstd(1.061, 0.0, 0.0);      // right_ankle;
        joint_info_[10].set_pos_actionstd(1.061, 0.0, 0.0);      // right_subtalar;
        joint_info_[11].set_pos_actionstd(1.061, 0.0, 0.0);      // right_mtp;
        joint_info_[12].set_pos_actionstd(2.0, 1.0, 2.0);        // left_hip;
        joint_info_[13].set_pos_actionstd(1.1, 0.0, 0.0);        // left_knee;
        joint_info_[14].set_pos_actionstd(1.061, 0.0, 0.0);      // left_ankle;
        joint_info_[15].set_pos_actionstd(1.061, 0.0, 0.0);      // left_subtalar;
        joint_info_[16].set_pos_actionstd(1.061, 0.0, 0.0);      // left_mtp;

        // actions for PD controller
        joint_info_[ 0].set_vel_actionmean(0.0, 0.0, 0.0);     // root_trs;
        joint_info_[ 1].set_vel_actionmean(0.0, 0.0, 0.0);     // root_rot;
        joint_info_[ 2].set_vel_actionmean(0.0, 0.0, 0.0);     // lumbar;
        joint_info_[ 3].set_vel_actionmean(0.0, 0.0, 0.0);     // right_shoulder;
        joint_info_[ 4].set_vel_actionmean(0.0, 0.0, 0.0);     // right_elbow;
        joint_info_[ 5].set_vel_actionmean(0.0, 0.0, 0.0);     // left_shoulder;
        joint_info_[ 6].set_vel_actionmean(0.0, 0.0, 0.0);     // left_elbow;
        joint_info_[ 7].set_vel_actionmean(0.0, 0.0, 0.0);     // right_hip;
        joint_info_[ 8].set_vel_actionmean(0.0, 0.0, 0.0);     // right_knee;
        joint_info_[ 9].set_vel_actionmean(0.0, 0.0, 0.0);     // right_ankle;
        joint_info_[10].set_vel_actionmean(0.0, 0.0, 0.0);     // right_subtalar;
        joint_info_[11].set_vel_actionmean(0.0, 0.0, 0.0);     // right_mtp;
        joint_info_[12].set_vel_actionmean(0.0, 0.0, 0.0);     // left_hip;
        joint_info_[13].set_vel_actionmean(0.0, 0.0, 0.0);     // left_knee;
        joint_info_[14].set_vel_actionmean(0.0, 0.0, 0.0);     // left_ankle;
        joint_info_[15].set_vel_actionmean(0.0, 0.0, 0.0);     // left_subtalar;
        joint_info_[16].set_vel_actionmean(0.0, 0.0, 0.0);     // left_mtp;

        joint_info_[ 0].set_vel_actionstd(0.0, 0.0, 0.0);        // root_trs;
        joint_info_[ 1].set_vel_actionstd(0.0, 0.0, 0.0);        // root_rot;
        joint_info_[ 2].set_vel_actionstd(10.0, 10.0, 10.0);     // lumbar;
        joint_info_[ 3].set_vel_actionstd(10.0, 10.0, 10.0);     // right_shoulder;
        joint_info_[ 4].set_vel_actionstd(10.0, 0.0, 0.0);       // right_elbow;
        joint_info_[ 5].set_vel_actionstd(10.0, 10.0, 10.0);     // left_shoulder;
        joint_info_[ 6].set_vel_actionstd(10.0, 0.0, 0.0);       // left_elbow;
        joint_info_[ 7].set_vel_actionstd(10.0, 10.0, 10.0);     // right_hip;
        joint_info_[ 8].set_vel_actionstd(10.0, 0.0, 0.0);       // right_knee;
        joint_info_[ 9].set_vel_actionstd(10.0, 0.0, 0.0);       // right_ankle;
        joint_info_[10].set_vel_actionstd(10.0, 0.0, 0.0);       // right_subtalar;
        joint_info_[11].set_vel_actionstd(10.0, 0.0, 0.0);       // right_mtp;
        joint_info_[12].set_vel_actionstd(10.0, 10.0, 10.0);     // left_hip;
        joint_info_[13].set_vel_actionstd(10.0, 0.0, 0.0);       // left_knee;
        joint_info_[14].set_vel_actionstd(10.0, 0.0, 0.0);       // left_ankle;
        joint_info_[15].set_vel_actionstd(10.0, 0.0, 0.0);       // left_subtalar;
        joint_info_[16].set_vel_actionstd(10.0, 0.0, 0.0);       // left_mtp;

        // actions for Torque controller
        joint_info_[ 0].set_torque_actionmean(0.0, 0.0, 0.0);     // root_trs;
        joint_info_[ 1].set_torque_actionmean(0.0, 0.0, 0.0);     // root_rot;
        joint_info_[ 2].set_torque_actionmean(0.0, 0.0, 0.0);     // lumbar;
        joint_info_[ 3].set_torque_actionmean(0.0, 0.0, 0.0);     // right_shoulder;
        joint_info_[ 4].set_torque_actionmean(0.0, 0.0, 0.0);     // right_elbow;
        joint_info_[ 5].set_torque_actionmean(0.0, 0.0, 0.0);     // left_shoulder;
        joint_info_[ 6].set_torque_actionmean(0.0, 0.0, 0.0);     // left_elbow;
        joint_info_[ 7].set_torque_actionmean(0.0, 0.0, 0.0);     // right_hip;
        joint_info_[ 8].set_torque_actionmean(0.0, 0.0, 0.0);     // right_knee;
        joint_info_[ 9].set_torque_actionmean(0.0, 0.0, 0.0);     // right_ankle;
        joint_info_[10].set_torque_actionmean(0.0, 0.0, 0.0);     // right_subtalar;
        joint_info_[11].set_torque_actionmean(0.0, 0.0, 0.0);     // right_mtp;
        joint_info_[12].set_torque_actionmean(0.0, 0.0, 0.0);     // left_hip;
        joint_info_[13].set_torque_actionmean(0.0, 0.0, 0.0);     // left_knee;
        joint_info_[14].set_torque_actionmean(0.0, 0.0, 0.0);     // left_ankle;
        joint_info_[15].set_torque_actionmean(0.0, 0.0, 0.0);     // left_subtalar;
        joint_info_[16].set_torque_actionmean(0.0, 0.0, 0.0);     // left_mtp;

        joint_info_[ 0].set_torque_actionstd(0.0, 0.0, 0.0);        // root_trs;
        joint_info_[ 1].set_torque_actionstd(0.0, 0.0, 0.0);        // root_rot;
        joint_info_[ 2].set_torque_actionstd(10.0, 10.0, 10.0);     // lumbar;
        joint_info_[ 3].set_torque_actionstd(10.0, 10.0, 10.0);     // right_shoulder;
        joint_info_[ 4].set_torque_actionstd(10.0, 0.0, 0.0);       // right_elbow;
        joint_info_[ 5].set_torque_actionstd(10.0, 10.0, 10.0);     // left_shoulder;
        joint_info_[ 6].set_torque_actionstd(10.0, 0.0, 0.0);       // left_elbow;
        joint_info_[ 7].set_torque_actionstd(10.0, 10.0, 10.0);     // right_hip;
        joint_info_[ 8].set_torque_actionstd(10.0, 0.0, 0.0);       // right_knee;
        joint_info_[ 9].set_torque_actionstd(10.0, 0.0, 0.0);       // right_ankle;
        joint_info_[10].set_torque_actionstd(10.0, 0.0, 0.0);       // right_subtalar;
        joint_info_[11].set_torque_actionstd(10.0, 0.0, 0.0);       // right_mtp;
        joint_info_[12].set_torque_actionstd(10.0, 10.0, 10.0);     // left_hip;
        joint_info_[13].set_torque_actionstd(10.0, 0.0, 0.0);       // left_knee;
        joint_info_[14].set_torque_actionstd(10.0, 0.0, 0.0);       // left_ankle;
        joint_info_[15].set_torque_actionstd(10.0, 0.0, 0.0);       // left_subtalar;
        joint_info_[16].set_torque_actionstd(10.0, 0.0, 0.0);       // left_mtp;

        joint_info_[ 0].set_pgain(0.0, 0.0, 0.0);       // root_trs;
        joint_info_[ 1].set_pgain(0.0, 0.0, 0.0);       // root_rot;
        joint_info_[ 2].set_pgain(60.0, 40.0, 60.0);    // lumbar;
        joint_info_[ 3].set_pgain(30.0, 30.0, 30.0);    // right_shoulder;
        joint_info_[ 4].set_pgain(40.0, 0.0, 0.0);      // right_elbow;
        joint_info_[ 5].set_pgain(30.0, 30.0, 30.0);    // left_shoulder;
        joint_info_[ 6].set_pgain(40.0, 0.0, 0.0);      // left_elbow;
        joint_info_[ 7].set_pgain(60.0, 40.0, 120.0);    // right_hip;
        joint_info_[ 8].set_pgain(120.0, 0.0, 0.0);      // right_knee;
        joint_info_[ 9].set_pgain(40.0, 0.0, 0.0);      // right_ankle;
        joint_info_[10].set_pgain(40.0, 0.0, 0.0);      // right_subtalar;
        joint_info_[11].set_pgain(40.0, 0.0, 0.0);      // right_mtp;
        joint_info_[12].set_pgain(60.0, 40.0, 120.0);    // left_hip;
        joint_info_[13].set_pgain(120.0, 0.0, 0.0);      // left_knee;
        joint_info_[14].set_pgain(40.0, 0.0, 0.0);      // left_ankle;
        joint_info_[15].set_pgain(40.0, 0.0, 0.0);      // left_subtalar;
        joint_info_[16].set_pgain(40.0, 0.0, 0.0);      // left_mtp;

        joint_info_[ 0].set_dgain(0.0, 0.0, 0.0);      // root_trs;
        joint_info_[ 1].set_dgain(0.0, 0.0, 0.0);      // root_rot;
        joint_info_[ 2].set_dgain(5.0, 5.0, 5.0);      // lumbar;
        joint_info_[ 3].set_dgain(5.0, 5.0, 5.0);      // right_shoulder;
        joint_info_[ 4].set_dgain(5.0, 0.0, 0.0);      // right_elbow;
        joint_info_[ 5].set_dgain(5.0, 5.0, 5.0);      // left_shoulder;
        joint_info_[ 6].set_dgain(5.0, 0.0, 0.0);      // left_elbow;
        joint_info_[ 7].set_dgain(5.0, 5.0, 5.0);      // right_hip;
        joint_info_[ 8].set_dgain(5.0, 0.0, 0.0);      // right_knee;
        joint_info_[ 9].set_dgain(5.0, 0.0, 0.0);      // right_ankle;
        joint_info_[10].set_dgain(5.0, 0.0, 0.0);      // right_subtalar;
        joint_info_[11].set_dgain(5.0, 0.0, 0.0);      // right_mtp;
        joint_info_[12].set_dgain(5.0, 5.0, 5.0);      // left_hip;
        joint_info_[13].set_dgain(5.0, 0.0, 0.0);      // left_knee;
        joint_info_[14].set_dgain(5.0, 0.0, 0.0);      // left_ankle;
        joint_info_[15].set_dgain(5.0, 0.0, 0.0);      // left_subtalar;
        joint_info_[16].set_dgain(5.0, 0.0, 0.0);      // left_mtp;

        joint_info_[ 0].set_ref_coord_index(1, 3, 2, -1);       // root_trs;
        joint_info_[ 1].set_ref_coord_index(4, 5, 6, 7);        // root_rot;
        joint_info_[ 2].set_ref_coord_index(8, 9, 10, 11);      // lumbar;
        joint_info_[ 3].set_ref_coord_index(20, 21, 22, 23);    // right_shoulder;
        joint_info_[ 4].set_ref_coord_index(24, -1, -1, -1);    // right_elbow;
        joint_info_[ 5].set_ref_coord_index(33, 34, 35, 36);    // left_shoulder;
        joint_info_[ 6].set_ref_coord_index(37, -1, -1, -1);    // left_elbow;
        joint_info_[ 7].set_ref_coord_index(12, 13, 14, 15);    // right_hip;
        joint_info_[ 8].set_ref_coord_index(16, -1, -1, -1);    // right_knee;
        joint_info_[ 9].set_ref_coord_index(17, -1, -1, -1);    // right_ankle;
        joint_info_[10].set_ref_coord_index(18, -1, -1, -1);    // right_subtalar;
        joint_info_[11].set_ref_coord_index(19, -1, -1, -1);    // right_mtp;
        joint_info_[12].set_ref_coord_index(25, 26, 27, 28);    // left_hip;
        joint_info_[13].set_ref_coord_index(29, -1, -1, -1);    // left_knee;
        joint_info_[14].set_ref_coord_index(30, -1, -1, -1);    // left_ankle;
        joint_info_[15].set_ref_coord_index(31, -1, -1, -1);    // left_subtalar;
        joint_info_[16].set_ref_coord_index(32, -1, -1, -1);    // left_mtp;
    }

    void ENVIRONMENT::set_obDouble_from_gc(Eigen::VectorXd &obDouble, Eigen::VectorXd &gc, int idx_begin)
    {
        // int idx_begin = 4; // ob_index for joint position begin from 4
        std::string joint_type;
        int ngv, gc_index[4];
        double v1[3], q1[4];

        for (int ijoint=2; ijoint<joint_info_.size(); ijoint++) {   // exclude root trs and rot
            joint_info_[ijoint].get_gc_index(gc_index);
            joint_type = joint_info_[ijoint].get_joint_type();
            ngv = joint_info_[ijoint].get_ngv();

            if (joint_type == "rot3") {
                q1[0] = gc(gc_index[0]);
                q1[1] = gc(gc_index[1]);
                q1[2] = gc(gc_index[2]);
                q1[3] = gc(gc_index[3]);
                Quaternion::quat2rotvec(v1, q1);
                obDouble(idx_begin+0) = v1[0];
                obDouble(idx_begin+1) = v1[1];
                obDouble(idx_begin+2) = v1[2];
            }
            if (joint_type == "rot1") {
                obDouble(idx_begin) = gc(gc_index[0]);
            }
            idx_begin += ngv;
        }
    }

    void ENVIRONMENT::get_endeffPos(Eigen::MatrixXd &gc_endeff) {
        raisim::Vec<3> pelvis_T;
        raisim::Mat<3, 3> pelvis_R{};
        anymal_->getBasePosition(pelvis_T);
        anymal_->getBaseOrientation(pelvis_R);
        raisim::Mat<3, 3> pelvis_R_inverse = pelvis_R.transpose();
        raisim::Vec<3> pelvis_T_inverse = {
                -pelvis_R_inverse[0] * pelvis_T[0] - pelvis_R_inverse[3] * pelvis_T[1] - pelvis_R_inverse[6] * pelvis_T[2],
                -pelvis_R_inverse[1] * pelvis_T[0] - pelvis_R_inverse[4] * pelvis_T[1] - pelvis_R_inverse[7] * pelvis_T[2],
                -pelvis_R_inverse[2] * pelvis_T[0] - pelvis_R_inverse[5] * pelvis_T[1] - pelvis_R_inverse[8] * pelvis_T[2]
        };

        // size_t idx_neck   = anymal_->getFrameIdxByName("neck");
        std::size_t idx_rwrist = anymal_->getFrameIdxByName("radius_hand_r");
        std::size_t idx_lwrist = anymal_->getFrameIdxByName("radius_hand_l");
        std::size_t idx_rankle = anymal_->getFrameIdxByName("ankle_r");
        std::size_t idx_lankle = anymal_->getFrameIdxByName("ankle_l");

        raisim::Vec<3> position;

        /// positions for r_wrist, l_wrist, r_ankle, l_ankle
        // anymal_->getFramePosition(idx_neck, position);
        // gc_endeff.row(0) << position(0), position(1), position(2);

        anymal_->getFramePosition(idx_rwrist, position);
        position = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, position);
        gc_endeff.row(0) << position(0), position(1), position(2);

        anymal_->getFramePosition(idx_lwrist, position);
        position = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, position);
        gc_endeff.row(1) << position(0), position(1), position(2);

        anymal_->getFramePosition(idx_rankle, position);
        position = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, position);
        gc_endeff.row(2) << position(0), position(1), position(2);

        anymal_->getFramePosition(idx_lankle, position);
        position = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, position);
        gc_endeff.row(3) << position(0), position(1), position(2);

        /// position for top_of_head
        raisim::Vec<3> pt_top_of_head; // top of head in torso coord
        pt_top_of_head[0] = 0.00105107; pt_top_of_head[1] = 0.663102; pt_top_of_head[2] = 0.;

        raisim::Mat<3, 3> torso_R{};
        raisim::Vec<3> torso_T;
        anymal_->getLink("torso").getPose(torso_T, torso_R);
        position = pt_coord_txf(torso_R, torso_T, pt_top_of_head);
        position = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, position);
        gc_endeff.row(4) << position(0), position(1), position(2);
    }

}

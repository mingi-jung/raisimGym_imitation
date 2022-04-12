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
//#include <raisim/OgreVis.hpp>
#include <raisim/RaisimServer.hpp>
#include "../../RaisimGymEnv.hpp"
//#include "visSetupCallback.hpp"

//#include "visualizer/raisimKeyboardCallback.hpp"
//#include "visualizer/helper.hpp"
//#include "visualizer/guiState.hpp"
//#include "visualizer/raisimBasicImguiPanel.hpp"

#include <nlohmann/json.hpp>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <GL/glut.h>
//#include <random>

using namespace std;

namespace raisim {

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
            read_ref_motion();

            /// add objects
            anymal_ = world_->addArticulatedSystem(resourceDir_ + "/urdf/MSK_GAIT2392_model_20201117.urdf");
            anymal_ref_ = world_->addArticulatedSystem(resourceDir_ + "/urdf/MSK_GAIT2392_model_20201117.urdf");
            anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            //ground_ = world_->addHeightMap(resourceDir_ + "/KAIST_terrain/ground.png", 0, 0, 60, 60, 0.0002, -6.547);
            //ground_ = world_->addHeightMap(resourceDir_ + "/KAIST_terrain/ground.png", 0, 0, 60, 60, 0.0002, -6.567);
            //ground_ = world_->addHeightMap(resourceDir_ + "/KAIST_terrain/ground.png", 0, 0, 60, 60, 0.0002, -6.561);
            ground_ = world_->addHeightMap(resourceDir_ + "/KAIST_terrain/ground.png", 0, 0, 60, 60, 0.0002, -6.559);
            //ground_ = world_->addHeightMap(resourceDir_ + "/KAIST_terrain/3_mingi.png", 34, 1.28, 80, 80, 0.00027, -8.848);
            //ground_ = world_->addHeightMap(resourceDir_ + "/KAIST_terrain/3_mingi.png", 34, 1.28, 80, 80, 0.00027, -8.8443);
            //ground_ = world_->addGround();
            // raisim::Ground* ground = world_->addGround();
            ground_->setName("Ground");
            world_->setERP(0, 0);
            /// get robot data
            gcDim_ = anymal_->getGeneralizedCoordinateDim(); //43
            gvDim_ = anymal_->getDOF(); //34
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

            //gc_ref_ = ref_motion_[0];
            /// this is nominal configuration of humanoid
            //gc_init_ = gc_ref_;
            // 0, 0, 0.87,                      // root translation
            // 0.7071068, 0.7071068, 0.0, 0.0,  // root rotation
            // 1, 0, 0, 0,                      // lumbar
            // 1, 0, 0, 0,                      // neck
            // 1, 0, 0, 0,                      // right shoulder
            // 0,                               // right elbow
            // 1, 0, 0, 0,                      // left shoulder
            // 0,                               // left elbow
            // 1, 0, 0, 0,                      // right hip
            // 0,                               // right knee
            // 1, 0, 0, 0,                      // right ankle
            // 1, 0, 0, 0,                      // left hip
            // 0,                               // left knee
            // 1, 0, 0, 0;                      // left ankle

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

            obDim_ = 86; /// convention described on top
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
                    0.5, //time_phase
                    Eigen::VectorXd::Constant(25, 0.0);
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
                    0.25, //time
                    Eigen::VectorXd::Constant(25, 0.3);

            //gui::rewardLogger.init({"angularPosReward", "angularVelReward", "endeffPosReward", "comPosReward"});

            /// indices of links that should not make contact with ground
            footIndices_.insert(anymal_->getBodyIdx("toes_r"));
            footIndices_.insert(anymal_->getBodyIdx("calcn_r"));
            footIndices_.insert(anymal_->getBodyIdx("toes_l"));
            footIndices_.insert(anymal_->getBodyIdx("calcn_l"));
            // footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
            // footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));



            /// visualize if it is the first environment
            //if (visualizable_) {
              server_ = std::make_unique<raisim::RaisimServer>(world_.get());
              server_->launchServer();
              server_->focusOn(anymal_);

            //}
        }

        ~ENVIRONMENT() final = default;

        void init() final {}

        void reset() final {

            double duration_ref_motion = cumultime_.back();
            //cout<<"reset_check"<<endl;
            // SKOO 20200630 Change it to a random number
            //time_ref_motion_start_ = 0.2 * duration_ref_motion;

            std::random_device rd; //random_device 생성
            std::mt19937 gen(rd()); //난수 생성 엔진 초기화
            std::uniform_int_distribution<int> dis(0,99); //0~99 까지 균등하게 나타나는 난수열 생성

            time_episode_ = ((double) dis(gen)/100) * duration_ref_motion;

            Eigen::VectorXd ref_motion_one_start(37);
            ref_motion_one_start.setZero();
            get_ref_motion(ref_motion_one_start, time_episode_);
            get_gv_init(gv_init_, time_episode_);

            anymal_->setState(ref_motion_one_start, gv_init_);
            time_phase_ = time_episode_/duration_ref_motion;

//            if (time_episode_ == 0) {
//                cout << "minngi" << endl;
//            }

            //anymal_->setState(gc_init_, gv_init_);

            updateObservation();
            // if (visualizable_)
            //     gui::rewardLogger.clean();
        }

        float step(const Eigen::Ref<EigenVec> &action) final {
          //cout<<"mingii1"<<endl;
            double v1[3], q1[4]; // temporary variables

            // std::cout << "action size : " << action.size() << std::endl;
            // std::cout << "action : " << action << std::endl;
            if (action.size() != actionDim_) {
                std::cout << "action size mismatch!!!" << std::endl;
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
            //cout<<"mingii2"<<endl;
            anymal_->setPdTarget(pTarget_, vTarget_);
            //cout<<"mingii3"<<endl;

            // dynamics simulation and visualization
            auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
            auto visDecimation = int(1. / (desired_fps_ * simulation_dt_) + 1e-10);
            Eigen::VectorXd ref_motion_one(37);
            double duration_ref_motion = cumultime_.back();
            int n_walk_cycle = 0; // mingi number of walking cycle
            double tau = 0;
            ref_motion_one.setZero();

            n_walk_cycle = (int) (time_episode_ / duration_ref_motion);
            tau = time_episode_ - n_walk_cycle * duration_ref_motion;
            //cout<<"mingii4"<<endl;
            get_ref_motion(ref_motion_one, tau); // it also considers the ref_motion time offset
            //cout<<"mingii5"<<endl;
            // ref body lateral position shift
            ref_motion_one[1] += 1.5;
            // ref body AP position shift
            // mingi, x translation increase because of wlking cycle
            ref_motion_one[0] += (ref_motion_.back()[0] - ref_motion_[0][0]) * n_walk_cycle; //- 0.05*n_walk_cycle;
            ref_motion_one[2] += (ref_motion_.back()[2] - ref_motion_[0][2]) * n_walk_cycle; //+ 0.03*n_walk_cycle;


            for (int i = 0; i < loopCount; i++) {
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();

                anymal_ref_->setGeneralizedCoordinate(ref_motion_one);

                // if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
                //     raisim::OgreVis::get()->renderOneFrame();
                if(server_) server_->unlockVisualizationServerMutex();
                visualizationCounter_++;
            }
            //cout<<"mingii6"<<endl;

            time_phase_ = tau/duration_ref_motion; //mingi, scaling of tau for observation
            // update gc_ and gv_
            //cout<<"mingii7"<<endl;
            updateObservation();
            //cout<<"mingii8"<<endl;

            anymal_ref_->getState(gc_ref_, gv_ref_);
            gc_ref_[1] = gc_ref_[1] - 1.5; // by mingi, make back again

            // First deepmimic reward: angular position error
            // w1*exp( -k1* sum(ang_pos_diff^2) )
            //cout<<"mingii9"<<endl;

            get_angularPosReward(difference_angle_, gc_, gc_ref_); // calculate the difference_angle_
            angularPosReward_ = angularPosRewardCoeff_ * difference_angle_;
            //cout<<"mingii10"<<endl;

            // Second deepmimic reward: angular velocity error
            // w2*exp( -k2* sum(ang_vel_diff^2) )
            // angularVelReward_ =  angularVelRewardCoeff_ * get_angularVelReward();
            //cout<<"mingii11"<<endl;
            // get_angularVelReward(difference_velocity_, gv_, tau);
            // angularVelReward_ = angularVelRewardCoeff_ * difference_velocity_;
            //cout<<"mingii12"<<endl;

            // Third deepmimic reward: end-effector position error
            // w3*exp( -K3* sum(endeff_pos_diff^2) )
            // endeffPosReward_ = endeffPosRewardCoeff_ * get_endeffPosReward();
            //cout<<"mingii13"<<endl;
            get_endeffPosReward(difference_end_);
            endeffPosReward_ = endeffPosRewardCoeff_ * difference_end_;
            //cout<<"mingii14"<<endl;

            // Fourth deepmimic reward: whole body COM error
            // w4*exp( -k4* com_diff^2 )
            // comPosReward_ = comPosRewardCoeff_ * get_comPosReward();

            raisim::Vec<3> com_ = anymal_->getCompositeCOM();
            raisim::Vec<3> com_ref_ = anymal_ref_->getCompositeCOM();
            com_ref_[1] -= 1.5;
            //cout<<"mingii15"<<endl;
            get_comPosReward(difference_com_, com_, com_ref_);
            comPosReward_ = comPosRewardCoeff_ * difference_com_;
            //cout<<"mingii16"<<endl;

            time_episode_ += (double) control_dt_;

            totalReward_ = angularPosReward_ + endeffPosReward_ + comPosReward_;

            if (isnan(totalReward_) != 0){
      				is_nan_detected_ = true;

      				angularPosReward_ = 0.;
      				//angularVelReward_ = 0.;
      				endeffPosReward_ = 0.;
      				comPosReward_ = 0.;
      				//forceReward_ = 0.;
      				totalReward_ = 0.;

      				//std::cout<<"NaN detected"<<std::endl;
      			}

            return totalReward_;
            // 0.65 * exp(-2 * difference_angle_); // + 0.1 * exp(-0.1 * difference_velocity/10),  0.15 * exp(-40 * difference_end_total) + 0.1 * exp(-10 * difference_end_chest)
        }


        void updateExtraInfo() final {
            //extraInfo_["forward vel reward"] = forwardVelReward_;
            extraInfo_["base height"] = gc_[2];
        }


        void updateObservation() {
            double v1[3], q1[4]; // temporary variables

            anymal_->getState(gc_, gv_);

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
            bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
            obDouble_.segment(29, 3) = bodyLinearVel_;
            obDouble_.segment(32, 3) = bodyAngularVel_;

            /// joint velocities
            obDouble_.segment(35, 25) = gv_.tail(25);
            obDouble_(60) = time_phase_;

            height_mat_.setZero(5,5);
            get_height(gc_, height_mat_);

            for (int i=0; i<5; i++){
              for (int j=0; j<5; j++){
                obDouble_(61 + 5*i + j) = height_mat_(i,j);
              }
            }

            //obScaled_ = (obDouble_ - obMean_).cwiseQuotient(obStd_);
        }


        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            //ob = obScaled_.cast<float>();
            ob = obDouble_.cast<float>();
        }


        bool isTerminalState(float &terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);

            /// 20200515 SKOO if a body other than feet has a contact with the ground
            //if (time_finish > 2){
            for (auto &contact: anymal_->getContacts()) {
                // std::cout << anymal_->getBodyNames()[contact.getlocalBodyIndex()] << " collides with: " << world_->getObject(contact.getPairObjectIndex())->getName() << std::endl;

                if (contact.getPairObjectIndex() == ground_->getIndexInWorld()) {
                    if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()) {
                        // std::cout << "A body other than the foot collides with Ground" << std::endl;

                        time_episode_ = 0;
                        time_phase_ = 0;

                        return true;
                    }
                }
            }

            if (is_nan_detected_ == true){
              is_nan_detected_ = false;

              time_episode_ = 0;
              time_phase_ = 0;

              return true;
            }

            if (gc_[0]*gc_[0] + gc_[1]*gc_[1] > 400){

              time_episode_ = 0;
              time_phase_ = 0;

              return true;

            }

            terminalReward = 0.f;
            return false;
        }


        void setSeed(int seed) final {
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

            // std::cout << v1[0] << ", " << v1[1] << ", " << v1[2] << std::endl;
            // std::cout << q1[0] << ", " << q1[1] << ", " << q1[2] << ", " << q1[3] << ", " << std::endl;
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

            // std::cout << v1[0] << ", " << v1[1] << ", " << v1[2] << std::endl;
            // std::cout << q1[0] << ", " << q1[1] << ", " << q1[2] << ", " << q1[3] << ", " << std::endl;
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


        template<typename Derived>
        inline bool is_finite(const Eigen::MatrixBase<Derived> &x) {
            return ((x - x).array() == (x - x).array()).all();
        }

        template<typename Derived>
        inline bool is_nan(const Eigen::MatrixBase<Derived> &x) {
            return ((x.array() == x.array())).all();
        }

        // prototypes of public functions
        void get_ref_motion(Eigen::VectorXd &, double);
        void read_ref_motion();
        void calc_slerp_joint(Eigen::VectorXd &, const int *, std::vector<Eigen::VectorXd> &, int, int, double);
        void calc_interp_joint(Eigen::VectorXd &, int , std::vector<Eigen::VectorXd> &, int, int, double);
        void get_angularPosReward(double &, Eigen::VectorXd &, Eigen::VectorXd &);
        void get_angularVelReward(double &, Eigen::VectorXd &, double);
        void get_comPosReward(double &, raisim::Vec<3>, raisim::Vec<3>);
        void get_endeffPosReward(double &);
        void get_gv_init(Eigen::VectorXd &, double);
        void get_height(Eigen::VectorXd &, Eigen::MatrixXd &);

    private:
        int gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        std::normal_distribution<double> distribution_;
        raisim::ArticulatedSystem *anymal_, *anymal_ref_;
        //raisim::Ground *ground_;
        raisim::HeightMap *ground_;
        //std::vector<GraphicObject> *anymalVisual_;
        //std::vector<GraphicObject> *anymalVisual_ref;
        Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget28_, vTarget_, torque_;
        Eigen::VectorXd gc_ref_, gv_ref_;
        double desired_fps_ = 60.;
        int visualizationCounter_ = 0;
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
        std::vector<Eigen::VectorXd> ref_motion_;
        std::string ref_motion_file_;
        std::vector<double> cumultime_;
        double time_ref_motion_start_ = 0.;
        double time_episode_ = 0.;
        double time_phase_ = 0.;

        double difference_angle_;
        double difference_velocity_;
        double difference_end_;
        double difference_com_;

        Eigen::MatrixXd height_mat_;

        bool is_nan_detected_ = false;

        //raisim::Vec<3> end_pos0, end_pos1, end_pos2, end_pos3, end_pos4, end_pos5, end_pos6, end_pos7;
    };


    void ENVIRONMENT::read_ref_motion() {
        Eigen::VectorXd jointNominalConfig(37);
        double totaltime = 0;
        //std::ifstream infile1(ref_motion_file_);
        std::ifstream infile1("/home/opensim2020/raisim_v3_workspace/raisimLib/raisimGymTorchBio/raisimGymTorch/env/envs/rsg_mingi/rsc/motions/mingi_walk_v2.txt");
        nlohmann::json  jsondata;
        infile1 >> jsondata;

        cumultime_.push_back(0);

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

            ref_motion_.push_back(jointNominalConfig);

            totaltime += (double) jsondata["Frames"][iframe][0];
            cumultime_.push_back(totaltime);
        }
        cumultime_.pop_back();
    }


    void ENVIRONMENT::get_ref_motion(Eigen::VectorXd &ref_motion_one, double tau) {
        int idx_frame_prev = -1;
        int idx_frame_next = -1;
        double t_offset;
        double t_offset_ratio;

        // SKOO 20200629 This loop should be checked again.
        for (int i = 1; i < cumultime_.size(); i++) { // mingi 20200706 checked
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


    void ENVIRONMENT::calc_interp_joint(
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

    void ENVIRONMENT::get_angularPosReward(
            double &difference_angle_,                       // output
            Eigen::VectorXd &gc_,
            Eigen::VectorXd &gc_ref_){

        double difference_quat = 0;
        double difference_rev = 0;

        for (int j = 0; j < 37; j++) { // all index of motion
            if (j == 7 || j == 11 || j == 16 || j == 21 || j == 29) {
                double qw1 = gc_ref_[j];
                Eigen::Vector3d qv1(gc_ref_[j + 1], gc_ref_[j + 2], gc_ref_[j + 3]);
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

                difference_rev = difference_rev + ((gc_ref_[j] - gc_[j]) * (gc_ref_[j] - gc_[j]));
            }

        }

        difference_angle_ = difference_quat + difference_rev;

        difference_angle_ = exp(-2 * difference_angle_);
    }

    void ENVIRONMENT::get_angularVelReward(
            double &difference_velocity_,                       // output
            Eigen::VectorXd &gv_,
            double tau){

        difference_velocity_ = 0; //mingi initialize
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
        Eigen::VectorXd jointIdx_quat_ref(5);
        Eigen::VectorXd jointIdx_euler_obs(5);

        jointIdx_quat_ref << //quaternion w idx
            7, 11,        // lumbar, right shoulder
            16, 21,       // left shoulder, right hip
            29;    // left hip
        jointIdx_euler_obs << //EULER idx
            6, 9,       // lumbar, right shoulder
            13, 17,       // left shoulder, right hip
            24;    // left hip
        for (int j = 0; j < 37; j++){
          for (int i = 0; i < 5; i++){ // jointIdx_quat_ref size
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

              if (qw4 > 0.99) {
                  qw4 = acos(0.99) * 2;
              }
              else {
                  qw4 = acos(qw4) * 2;
              }

              difference_velocity_ += pow(sqrt(pow(qw4/small_dt, 2)) - sqrt(pow(gv_[jointIdx_euler_obs[i]], 2) + pow(gv_[jointIdx_euler_obs[i]+1], 2) + pow(gv_[jointIdx_euler_obs[i]+2],2)), 2);


            }
          }

          if (j == 15 || j == 20 || j == 25 || j == 26 || j == 27 || j == 28 || j == 33 || j == 34 || j == 35 || j == 36){ //rev joints
              if (j == 15) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[12], 2)), 2);
              }
              if (j == 20) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[16], 2)), 2);
              }
              if (j == 25) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[20], 2)), 2);
              }
              if (j == 26) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[21], 2)), 2);
              }
              if (j == 27) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[22], 2)), 2);
              }
              if (j == 28) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[23], 2)), 2);
              }
              if (j == 33) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[27], 2)), 2);
              }
              if (j == 34) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[28], 2)), 2);
              }
              if (j == 35) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[29], 2)), 2);
              }
              if (j == 36) {
                  difference_velocity_ +=
                          pow(sqrt(pow((ref_motion_one_next[j] - ref_motion_one_prev[j]) / small_dt, 2)) - sqrt(pow(gv_[30], 2)), 2);
              }
          }

        }
        cout<<"mmingi5"<<endl;
        //difference_velocity_ = exp(-0.1 * difference_velocity_);
        difference_velocity_ = exp(-0.00004 * difference_velocity_);


    }

    void ENVIRONMENT::get_comPosReward(
            double &difference_com_,                       // output
            raisim::Vec<3> com_,
            raisim::Vec<3> com_ref_){

            // difference_com_ = pow((com_[0] - gc_[0]) - (com_ref_[0] - gc_ref_[0]),2) + pow((com_[1] - gc_[1]) - (com_ref_[1] - gc_ref_[1]),2) + pow((com_[2] - gc_[2]) - (com_ref_[2] - gc_ref_[2]),2);
            // difference_com_ = exp(-10 * difference_com_);

            // local com
            difference_com_ = pow(com_[2] - com_ref_[2],2);
            difference_com_ = exp(-30 * difference_com_);
    }

    void ENVIRONMENT::get_endeffPosReward(
            double &difference_end_){

            raisim::Vec<3> end_pos0, end_pos1, end_pos2, end_pos3, end_pos4, end_pos5, end_pos6, end_pos7, end_root1, end_root2;

            anymal_->getFramePosition(0, end_root1);
            anymal_->getFramePosition(8, end_pos0);
            anymal_->getFramePosition(11, end_pos1);
            anymal_->getFramePosition(14, end_pos2);
            anymal_->getFramePosition(19, end_pos3);

            anymal_ref_->getFramePosition(0, end_root2);
            end_root2[1] -= 1.5;
            anymal_ref_->getFramePosition(8, end_pos4);
            end_pos4[1] -= 1.5;
            anymal_ref_->getFramePosition(11, end_pos5);
            end_pos5[1] -= 1.5;
            anymal_ref_->getFramePosition(14, end_pos6);
            end_pos6[1] -= 1.5;
            anymal_ref_->getFramePosition(19, end_pos7);
            end_pos7[1] -= 1.5;

            difference_end_ = 0;
            difference_end_ += pow((end_pos0[0]-end_root1[0]) - (end_pos4[0]-end_root2[0]),2) + pow((end_pos0[1]-end_root1[1]) - (end_pos4[1]-end_root2[1]),2) + pow((end_pos0[2]-end_root1[2]) - (end_pos4[2]-end_root2[2]),2);
            difference_end_ += pow((end_pos1[0]-end_root1[0]) - (end_pos5[0]-end_root2[0]),2) + pow((end_pos1[1]-end_root1[1]) - (end_pos5[1]-end_root2[1]),2) + pow((end_pos1[2]-end_root1[2]) - (end_pos5[2]-end_root2[2]),2);
            difference_end_ += pow((end_pos2[0]-end_root1[0]) - (end_pos6[0]-end_root2[0]),2) + pow((end_pos2[1]-end_root1[1]) - (end_pos6[1]-end_root2[1]),2) + pow((end_pos2[2]-end_root1[2]) - (end_pos6[2]-end_root2[2]),2);
            difference_end_ += pow((end_pos3[0]-end_root1[0]) - (end_pos7[0]-end_root2[0]),2) + pow((end_pos3[1]-end_root1[1]) - (end_pos7[1]-end_root2[1]),2) + pow((end_pos3[2]-end_root1[2]) - (end_pos7[2]-end_root2[2]),2);

            difference_end_ = exp(-40 * difference_end_);
    }

    void ENVIRONMENT::get_gv_init(
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

}

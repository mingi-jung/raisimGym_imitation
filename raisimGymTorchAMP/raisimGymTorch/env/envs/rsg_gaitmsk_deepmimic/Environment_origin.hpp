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
* URDF Model: MSK_GAIT2392_model_20201214.urdf
* general coordinate indices
* root_trs        trs3      3   3       0,1,2
* root_rot        spherical 4   3       3,4,5,6
* back            spherical 4   3       7,8,9,10
* acromial_r      spherical 4   3       11,12,13,14
* elbow_r         revolute  1   1       15
* acromial_l      spherical 4   3       16,17,18,19
* elbow_r         revolute  1   1       20
* hip_r           spherical 4   3       21,22,23,24
* walker_knee_r   revolute  1   1       25
* ankle_r         revolute  1   1       26
* subtalar_r      revolute  1   1       27
* mtp_r           revolute  1   1       28
* hip_l           spherical 4   3       29,30,31,32
* walker_knee_l   revolute  1   1       33
* ankle_l         revolute  1   1       34
* subtalar_l      revolute  1   1       35
* mtp_l           revolute  1   1       36
*                          37  31
*
*   action space = [ muscles        92
*                    lumbar          3
*                    right shoulder  3
*                    right elbow     1
*                    left shoulder   3
*                    left elbow      1 ]  total 103
*
*   observation space = [ height                                                      n =  1, si =  0
*                         z-axis of body in the world frame expressed                 n =  3, si =  1
*                         joint angles (gc),                                          n = 25, si =  4
*                         body Linear velocities,                                     n =  3, si = 29
*                         body Angular velocities,                                    n =  3, si = 32
*                         joint velocities (gv),                                      n = 25, si = 35
*                         grf and cop                                                 n = 12, si = 60
*                         target speed                                                n =  1, si = 72
*                         target direction                                            n =  3, si = 73 ] 76
*/


#pragma once

#include <cstdlib>
#include <set>
#include <chrono>
#include "../../RaisimGymEnv.hpp"
#include "DeepmimicUtility.hpp"
#include "Muscle.hpp"


namespace raisim {
    class ENVIRONMENT : public RaisimGymEnv {
    public:
        explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
                RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable) {

            Muscles_.resize(92);
            muscle_trs_.resize(16);
            muscle_rot_.resize(16);
            scale_link_.resize(20);
            scale_value_.resize(20);
            JointTorque_muscle_.resize(25); std::fill(JointTorque_muscle_.begin(), JointTorque_muscle_.end(), 0.);
            JointTorque_direct_.resize(25); std::fill(JointTorque_direct_.begin(), JointTorque_direct_.end(), 0.);
            JointTorque_pdcont_.resize(25); std::fill(JointTorque_pdcont_.begin(), JointTorque_pdcont_.end(), 0.);

            /// Reward coefficients
            READ_YAML(double, angularPosRewardCoeff_, cfg["angularPosRewardCoeff"])
            READ_YAML(double, angularVelRewardCoeff_, cfg["angularVelRewardCoeff"])
            READ_YAML(double, endeffPosRewardCoeff_, cfg["endeffPosRewardCoeff"])
            READ_YAML(double, comPosRewardCoeff_, cfg["comPosRewardCoeff"])

            /// Step info flags
            READ_YAML(int, info_train_info_on_, cfg["train_info_on"])

            /// curriculum learning
            READ_YAML(int, curriculumPhase_, cfg["curriculum_phase"])
            curriculumFactor_ = 0.0;

            /// add objects
            std::string URDF_filename = resourceDir_ + "/urdf/MSK_GAIT2392_model_subject06.urdf";
            anymal_ = world_->addArticulatedSystem(URDF_filename);
            anymal_->setName("anymal");

            /// load urdf-specific joint information
            GetJointInfo();

            anymal_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);
            // anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);

            ground_ = world_->addGround();
            // raisim::Sphere *sphere_x_axis = world_->addSphere(0.2, 1.0, "red");
            // raisim::Sphere *sphere_y_axis = world_->addSphere(0.2, 1.0, "green");
            // sphere_x_axis->setPosition(30, 0, 0.3);
            // sphere_y_axis->setPosition(0, 30, 0.3);

            /// DeepMimic; a robot to show the reference motion
            world_ref_.setTimeStep(1e-10);
            anymal_ref_ = world_ref_.addArticulatedSystem(URDF_filename);

            /// get robot data
            gcDim_ = anymal_->getGeneralizedCoordinateDim();  // 37
            gvDim_ = anymal_->getGeneralizedVelocityDim();    // 31

            nJoints_ = gvDim_ - 6; // 31-6 = 25
            nMuscles_ = 92;

            /// MUST BE DONE FOR ALL ENVIRONMENTS
            obDim_ = 60 + 12 + 4; // convention described on top
            actionDim_ = nMuscles_ + nJoints_ + nJoints_; // 25(PD) + 25(Torque)
            obDim_amp_ = 71;

            /// initialize containers
            gc_.setZero(gcDim_);
            gv_.setZero(gvDim_);
            PdTarget_gc_.setZero(gcDim_); PdTarget_gv_.setZero(gvDim_); pTargetAction_.setZero(actionDim_);

            /// set pd gains
            Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
            set_jointPgain_jointDgain(jointPgain, jointDgain);
            anymal_->setPdGains(jointPgain, jointDgain);
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// indices of links that should not make contact with ground
            footIndices_.insert(anymal_->getBodyIdx("toes_r"));
            footIndices_.insert(anymal_->getBodyIdx("calcn_r"));
            footIndices_.insert(anymal_->getBodyIdx("toes_l"));
            footIndices_.insert(anymal_->getBodyIdx("calcn_l"));

            /// action scaling parameters
            actionMean_.setZero(actionDim_);
            actionStd_.setZero(actionDim_);

            jointPosActionMean_.setZero(nJoints_);
            jointPosActionStd_.setZero(nJoints_);
            set_joint_pos_action_scales(jointPosActionMean_, jointPosActionStd_);

            /*
            jointVelActionMean_.setZero(nJoints_);
            jointVelActionStd_.setZero(nJoints_);
            set_joint_vel_action_scales(jointVelActionMean_, jointVelActionStd_); */

            jointTorqueActionMean_.setZero(nJoints_);
            jointTorqueActionStd_.setZero(nJoints_);
            set_joint_torque_action_scales(jointTorqueActionMean_, jointTorqueActionStd_);

            actionMean_.segment(0, nMuscles_) = Eigen::VectorXd::Constant(nMuscles_, 0.3);
            actionMean_.segment(nMuscles_, nJoints_) = jointPosActionMean_;
            // actionMean_.segment(nMuscles_ + nJoints_, nJoints_) = jointVelActionMean_;
            actionMean_.segment(nMuscles_ + nJoints_, nJoints_) = jointTorqueActionMean_;

            actionStd_.segment(0, nMuscles_) = Eigen::VectorXd::Constant(nMuscles_, 0.3);
            actionStd_.segment(nMuscles_, nJoints_) = jointPosActionStd_;
            // actionStd_.segment(nMuscles_ + nJoints_, nJoints_) = jointVelActionStd_;
            actionStd_.segment(nMuscles_ + nJoints_, nJoints_) = jointTorqueActionStd_;

            /// Deepmimic information
            deepmimic_.set_joint_info(joint_info_);  // copy the same information to deepmimic operators
            deepmimic_.set_gcdim_gvdim_njoints(gcDim_, gvDim_, nJoints_); // 37, 31, 25

            /// DeepMimic; read a sequence of reference motion to mimic
            std::string ref_motion_file;
            std::vector<std::string> ref_motion_filenames;
            READ_YAML(int, num_ref_motion_, cfg["num_ref_motion"])

            for (int irefmotion=0; irefmotion<num_ref_motion_; irefmotion++) {
                READ_YAML(std::string, ref_motion_file, cfg["ref_motion_file_"+std::to_string(irefmotion+1)])
                ref_motion_filenames.push_back(ref_motion_file);
            }

            for (int irefmotion=0; irefmotion<num_ref_motion_; irefmotion++) {
                deepmimic_.set_current_ref_motion_index(irefmotion);
                deepmimic_.add_ref_motion(resourceDir_ + "/refmotion/" + ref_motion_filenames[irefmotion]);
                deepmimic_.set_ref_motion_lateral_offset(gc_ref_body_y_offset_);  // shift the reference model
            }

            visualizable_ = true; // SKOO 20200109 temporarily
            if (visualizable_) {
                server_ = std::make_unique<raisim::RaisimServer>(world_.get());
                server_->launchServer();
                server_->focusOn(anymal_);
            }

            // SKOO 20210408 add a ghost and set its color
            // anymal_vis_ = server_->addVisualArticulatedSystem("anymal_vis", URDF_filename);
            // anymal_vis_->color = { 0.3, 0.0, 0.0, 0.9 };

            ReadMuscleModel(resourceDir_ +  "/osim2musclejson/2392_20201124.txt");
            double muscle_initial_length[92];
            ReadMuscleInitialLength(resourceDir_ +  "/osim2musclejson/Muscle_Initial_Length.json", muscle_initial_length);
            update_seg_txf();
            GetScaleFactor(URDF_filename);
            for (int imuscles=0; imuscles<nMuscles_; imuscles++){
                Muscles_[imuscles].model = anymal_;
                Muscles_[imuscles].m_control_dt = control_dt_;
                Muscles_[imuscles].initialize(muscle_trs_, muscle_rot_);
                Muscles_[imuscles].HillTypeOn = 1;
                Muscles_[imuscles].SetScale(scale_link_, scale_value_, muscle_trs_, muscle_rot_, muscle_initial_length[imuscles]);
                Muscles_[imuscles].Fiso *= 2.0;  // increase the maximum muscle strength by factor of two
            }

            for (int imuscles=0; imuscles<nMuscles_; imuscles++){
                Muscles_[imuscles].VisOn(*server_);
            }

            Right_contact_ = server_->addVisualPolyLine("Right_contact");
            Left_contact_ = server_->addVisualPolyLine("Left_contact");
            Right_contact_->color[0] = 0.; Right_contact_->color[1] = 1.; Right_contact_->color[2] = 0.;
            Left_contact_->color[0] = 0.; Left_contact_->color[1] = 1.; Left_contact_->color[2] = 0.;
        }

        void init() final {}

        void reset() final {
            /// set the reference motion index
            deepmimic_.set_current_ref_motion_random();

            /// Random start of simulation
            tau_random_start_ = deepmimic_.get_tau_random_start();

            for (int iMuscle = 0; iMuscle < nMuscles_; iMuscle++) {
                Muscles_[iMuscle].Activation = -0.01; // See Muscle.hpp Line# 745
                // Muscles[iMuscle].FiberLength = Muscles[iMuscle].OptFiberLength;
            }

            Eigen::VectorXd gc0(gcDim_), gc1(gcDim_), gc2(gcDim_), gv0(gvDim_);
            deepmimic_.get_gc_ref_motion(gc0, tau_random_start_);
            deepmimic_.get_gc_ref_motion(gc1, tau_random_start_ - 1*simulation_dt_);
            deepmimic_.get_gc_ref_motion(gc2, tau_random_start_ + 1*simulation_dt_);
            deepmimic_.get_gv_ref_motion(gv0, gc1, gc2, 2*simulation_dt_);

            /// DeepMimic; set the reference motion to the kinematics model
            Eigen::VectorXd gc_init; gc_init.setZero(gcDim_);
            Eigen::VectorXd gv_init; gv_init.setZero(gvDim_);

            anymal_ref_->setState(gc0, gv0);
            world_ref_.integrate();

            gc_init = gc0;  // Eigen::VectorXd copy
            gc_init(1) += gc_ref_body_y_offset_;  // ref_motion shifted when it is loaded.
            gv_init = gv0;   // Eigen::VectorXd copy

            anymal_->setState(gc_init, gv_init); // We want to put the simulation model at y = 0
            simulationCounter_ = 0;

            /// Muscle geometry update
            update_seg_txf();
            for (int imuscles = 0; imuscles<nMuscles_; imuscles++) {
                Muscles_[imuscles].UpdateGlobalPos(muscle_trs_, muscle_rot_);
                Muscles_[imuscles].VisUpdate(*server_);
            }

            /// AMP goals
            target_speed_ = 1.05; // for this specific refmotion data
            target_direction_ << 1, 0, 0;  // direction in the root local coordinate

            /// debugging flag
            is_nan_detected_ = false;

            updateObservation();
        }

        float step(const Eigen::Ref<EigenVec> &action) final {
            /// time start
            auto time0 = std::chrono::steady_clock::now();

            /// Action scaling
            pTargetAction_ = action.cast<double>();
            pTargetAction_ = pTargetAction_.cwiseProduct(actionStd_);
            pTargetAction_ += actionMean_;

            for (int iJoint = 0; iJoint < nJoints_; iJoint++) {
                JointTorque_muscle_[iJoint] = 0.0; // 20210331 SKOO I would like to calculate 25 joint actuation torques by pd control
                JointTorque_pdcont_[iJoint] = 0.0; // 20210331 SKOO I would like to calculate 25 joint actuation torques by pd control
                JointTorque_direct_[iJoint] = 0.0; // 20210331 SKOO I would like to calculate 25 joint actuation torques by pd control
            }

            /// joint actuation by pd control
            PdTarget_gc_.setZero(gcDim_); PdTarget_gv_.setZero(gvDim_);

            Eigen::VectorXd pTargetAction_temp = pTargetAction_.segment(nMuscles_, nJoints_);          // 92 - 116 (25)
            int num_gc_temp = set_gc_from_action(PdTarget_gc_, pTargetAction_temp);  // action(25) into gc(7+30)
            // if (num_gc_temp != gcDim_-7)
            //     std::cout << "Error @ CheckPoint S1!" << std::endl;

            // We replace setPdTarget with our own stable PD controller
            anymal_->setPdTarget(PdTarget_gc_, PdTarget_gv_);

            /*
            anymal_->getState(gc_, gv_);

            std::string joint_type;  // 20210331 SKOO see stable PD control by Tan el al., 2011
            int gc_index[4], gv_index[3];
            double kp[3], kd[3];

            for (int ijoint=2; ijoint<joint_info_.size(); ijoint++) { // exclude the first two root joints to calculate only the 25 actuators
                joint_info_[ijoint].get_gc_index(gc_index);
                joint_info_[ijoint].get_gv_index(gv_index);
                joint_info_[ijoint].get_pgain(kp);
                joint_info_[ijoint].get_dgain(kd);
                joint_type = joint_info_[ijoint].get_joint_type();

                // SKOO 20210331 suppress the Kd-term for now
                // I could not make the Kd-term work
                kd[0] = 0.0; kd[1] = 0.0; kd[2] = 0.0;

                if (joint_type == "rot3") {
                    double quat1[4], quatbar[4]; // qdot1[3], qddot1[3], qdotbar[3];
                    double vq[3]; // log_quat_diff vector

                    quatbar[0] = PdTarget_gc_[gc_index[0]]; // W
                    quatbar[1] = PdTarget_gc_[gc_index[1]]; // X
                    quatbar[2] = PdTarget_gc_[gc_index[2]]; // Y
                    quatbar[3] = PdTarget_gc_[gc_index[3]]; // Z

                    quat1[0] = gc_[gc_index[0]];
                    quat1[1] = gc_[gc_index[1]];
                    quat1[2] = gc_[gc_index[2]];
                    quat1[3] = gc_[gc_index[3]];

                    Quaternion::get_log_quat_diff(vq, quat1, quatbar);

                    // - kp * (-1*ln(q1.inv * qbar) + control_dt_*qdot1)
                    // - kd * (qdot1 + control_dt_*(qdot2-qdot1)/dt_ - qdotbar) // stable pd controller in Tan et al., 2011

                    for (int ii=0; ii<3; ii++) {
                        double p_temp = (-2 * vq[ii] + control_dt_ * gv_[gv_index[ii]]);
                        // double d_temp = (gv_[gv_index[ii]] + (gv_[gv_index[ii]] - gv_pre_[gv_index[ii]]) - PdTarget_gv_[gv_index[ii]]);
                        // JointTorque_pdcont_[gv_index[ii] - 6] = - kp[ii] * p_temp - kd[ii] * d_temp;
                        // 20210401 SKOO I could not make the Kd-term work
                        JointTorque_pdcont_[gv_index[ii] - 6] = - kp[ii] * p_temp;
                    }
                    // std::cout << "rot3, ";
                    // std::cout << kp[0] << ", " << kp[1] << ", " << kp[2] << ", ";
                    // std::cout << kd[0] << ", " << kd[1] << ", " << kd[2] << ", ";
                    // std::cout << gv_[gv_index[0]] << ", " << gv_[gv_index[1]] << ", " << gv_[gv_index[2]] << ", ";
                    // std::cout << gv_pre_[gv_index[0]] << ", " << gv_pre_[gv_index[1]] << ", " << gv_pre_[gv_index[2]] << ",,,,,,, ";
                    // std::cout << gv_index[0] << ", " << gv_index[1] << ", " << gv_index[2] << ", ";
                    // std::cout << JointTorque_pdcont_[gv_index[0]-6] << ", " << JointTorque_pdcont_[gv_index[1]-6] << ", " << JointTorque_pdcont_[gv_index[2]-6] << std::endl;
                    // std::cout << "done" << std::endl;
                }
                if (joint_type == "rot1") {
                    JointTorque_pdcont_[gv_index[0]-6] =
                            - kp[0]*(gc_[gc_index[0]] + control_dt_*gv_[gv_index[0]] - PdTarget_gc_[gc_index[0]])
                            - kd[0]*(gv_[gv_index[0]] + (gv_[gv_index[0]] - gv_pre_[gv_index[0]]) - PdTarget_gv_[gv_index[0]]);
                    // std::cout << "rot1, ";
                    // std::cout << JointTorque_pdcont_[gv_index[0]] << ", ";
                    // std::cout << "done" << std::endl;
                }
                if (joint_type == "trs3") {
                    for (int ii=0; ii<3; ii++)
                        JointTorque_pdcont_[gv_index[ii]-6] =
                            - kp[ii]*(gc_[gc_index[ii]] + control_dt_*gv_[gv_index[ii]] - PdTarget_gc_[gc_index[ii]])
                            - kd[ii]*(gv_[gv_index[ii]] + (gv_[gv_index[ii]] - gv_pre_[gv_index[ii]]) - PdTarget_gv_[gv_index[ii]]);
                    // std::cout << "trs3, ";
                    // std::cout << JointTorque_pdcont_[gv_index[0]] << ", " << JointTorque_pdcont_[gv_index[1]] << ", " << JointTorque_pdcont_[gv_index[2]] << ", ";
                    // std::cout << "done" << std::endl;
                }
            }

            // std::cout << "Error @ CheckPoint S3!" << std::endl;
            gv_pre_ = gv_; // It is used to calculate the kd-term

            /// joint actuation by muscles
            if (curriculumPhase_ == 1 || curriculumPhase_ == 2) {
                update_seg_txf();
                std::vector<double> JointTorque_muscle_temp(31, 0.);
                for (int iMuscle = 0; iMuscle < nMuscles_; iMuscle++) {  // 0 - 91
                    Muscles_[iMuscle].excitation = pTargetAction_[iMuscle];
                    Muscles_[iMuscle].Update(muscle_trs_, muscle_rot_, JointTorque_muscle_temp); // resultant joint torques
                }
                for (int iJoint = 0; iJoint < nJoints_; iJoint++)
                    JointTorque_muscle_[iJoint] = JointTorque_muscle_temp[6 + iJoint];
            }

            /// joint actuation direct
            for (int iJoint = 0; iJoint < nJoints_; iJoint++)
                JointTorque_direct_[iJoint] = pTargetAction_[nMuscles_ + nJoints_ + iJoint];

            // curriculum transition by selective application of different torques
            if (curriculumPhase_ == 0) {
                for (int iJoint = 0; iJoint < nJoints_; iJoint++) {
                    // SKOO 20210401 We do not use the direct torque control of joints
                    JointTorque_direct_[iJoint] = 0.0;
                    // SKOO 20210401 We do not use the muscle control in curriculum phase 0
                    JointTorque_muscle_[iJoint] = 0.0;
                }
            }

            if (curriculumPhase_ == 1) {
                for (int iJoint = 0; iJoint < nJoints_; iJoint++) {
                    // SKOO 20210401 We do not use the direct torque control of joints
                    JointTorque_direct_[iJoint] = 0.0;

                    double JointTorqueMax[25] = {50, 40, 60,       // lumbar
                                                 40, 20, 30, 30,   // right shoulder and elbow
                                                 40, 20, 30, 30,   // left shoulder and elbow
                                                 150, 60, 100, 130, 150, 40, 30,
                                                 150, 60, 100, 130, 150, 40, 30};

                    if (iJoint < 3 || iJoint > 10) {  // we use the pd control for the shoulder and elbow
                        // SKOO 20210401 decrease the torque_by_pdcontrol gradually
                        if (JointTorque_pdcont_[iJoint] > (1 - curriculumFactor_) * (1.5) * JointTorqueMax[iJoint])
                            JointTorque_pdcont_[iJoint] = (1 - curriculumFactor_) * (1.5) * JointTorqueMax[iJoint];
                        if (JointTorque_pdcont_[iJoint] < (1 - curriculumFactor_) * (-1.5) * JointTorqueMax[iJoint])
                            JointTorque_pdcont_[iJoint] = (1 - curriculumFactor_) * (-1.5) * JointTorqueMax[iJoint];
                    }

                    // SKOO 20210401 increase the torque_by_muscle gradually
                    if (JointTorque_muscle_[iJoint] > curriculumFactor_ * (1.5) * JointTorqueMax[iJoint])
                        JointTorque_muscle_[iJoint] = curriculumFactor_ * (1.5) * JointTorqueMax[iJoint];
                    if (JointTorque_muscle_[iJoint] < curriculumFactor_ * (-1.5) * JointTorqueMax[iJoint])
                        JointTorque_muscle_[iJoint] = curriculumFactor_ * (-1.5) * JointTorqueMax[iJoint];
                }
            }

            if (curriculumPhase_ == 2) {  // pure muscle control for lower limbs
                for (int iJoint = 0; iJoint < nJoints_; iJoint++) {
                    // SKOO 20210401 We do not use the direct torque control of joints
                    JointTorque_direct_[iJoint] = 0.0;

                    if (iJoint < 3 || iJoint > 10)  // we use the pd control for the shoulder and elbow
                        JointTorque_pdcont_[iJoint] = 0.0;
                }
            }

            /// application of the actuation torques to joint actuators
            raisim::VecDyn JointTorque = anymal_->getFeedForwardGeneralizedForce(); // gvDim_
            for (int iJoint = 0; iJoint < 6; iJoint++)
                JointTorque[iJoint] = 0.0;  // input zeros for base body actuations

            for (int iJoint = 0; iJoint < nJoints_; iJoint++) {
                JointTorque[6 + iJoint] = JointTorque_pdcont_[iJoint]
                        + JointTorque_muscle_[iJoint] + JointTorque_direct_[iJoint];
            }

            anymal_->setGeneralizedForce(JointTorque);
            */

            /// timers to measure the time for reward calculation
            auto time1 = std::chrono::steady_clock::now();
            time_part1_ = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0).count();

            /// DeepMimic; gc and gv initialization
            Eigen::VectorXd gc_ref, gv_ref;
            gc_ref.setZero(gcDim_);
            gv_ref.setZero(gvDim_);

            /// DeepMimic; set reference motion
            double tau; // 20200706 time in simulation
            tau = tau_random_start_ + simulationCounter_ * simulation_dt_;
            deepmimic_.get_gc_ref_motion(gc_ref, tau);
            anymal_ref_->setState(gc_ref, gv_ref);  // 20201208 SKOO Need to be fixed for right visualization
            world_ref_.integrate();

            /// visualization in the unity
            int n_integrate = int(control_dt_ / simulation_dt_ + 1e-10);
            for (int i=0; i<n_integrate; i++) {
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();
                simulationCounter_++;   // 20201006 SKOO We need it to play the reference motion
                // if (i == n_integrate - 2) {
                //     anymal_->getState(gc_pre_, gv_pre_);
                //     std::cout << "gv_pre_ : " << gv_pre_ << std::endl;
                // }

                // anymal_ref_->setGeneralizedCoordinate(ref_motion_one);
                if(server_) server_->unlockVisualizationServerMutex();
            }

            update_seg_txf();
            for (int imuscles = 0; imuscles<nMuscles_; imuscles++) {
                Muscles_[imuscles].UpdateGlobalPos(muscle_trs_, muscle_rot_);
                Muscles_[imuscles].VisUpdate(*server_);
            }

            auto time2 = std::chrono::steady_clock::now();
            time_part2_ = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();

            /// Update obDouble_ using gc_ and gv_ (no phase)
            updateObservation();

            /// Reward calculation
            tau = tau_random_start_ + simulationCounter_ * simulation_dt_;
            deepmimic_.get_gc_ref_motion(gc_ref, tau);

            Eigen::VectorXd gc_ref_1(gcDim_), gc_ref_2(gcDim_);
            deepmimic_.get_gc_ref_motion(gc_ref_1, tau - 1*simulation_dt_);
            deepmimic_.get_gc_ref_motion(gc_ref_2, tau + 1*simulation_dt_);
            deepmimic_.get_gv_ref_motion(gv_ref, gc_ref_1, gc_ref_2, 2*simulation_dt_);
            anymal_ref_->setState(gc_ref, gv_ref);
            world_ref_.integrate(); // to update center of mass information
            // anymal_vis_->setGeneralizedCoordinate(gc_ref); // ghost is updated here because gc_ref is available here

            // First deepmimic reward: angular position error by MG and Prof.
            double sumsquare_diffpos = deepmimic_.get_angularPosReward(gc_, gc_ref);
            double pose_reward = exp(-1.0 * 0.40 * sumsquare_diffpos);
            angularPosReward_ = angularPosRewardCoeff_ * pose_reward;
            angularPosReward_raw_ = sqrt(0.40 * sumsquare_diffpos);

            // Second deepmimic reward: angular velocity error by Prof.
            double sumsquare_diffvel = deepmimic_.get_angularVelReward(gv_, gv_ref);
            double vel_reward = exp(-1.0 * 0.000625 * sumsquare_diffvel);
            angularVelReward_ =  angularVelRewardCoeff_ * vel_reward;
            angularVelReward_raw_ = sqrt(0.000625 * sumsquare_diffvel);

            // Third deepmimic reward: end-effector position error by MG
            // head, hands, feet
            update_seg_txf(); // by YJKOO
            int n_endeff = 5;
            Eigen::MatrixXd gc_endeff(n_endeff, 3), gc_ref_endeff(n_endeff, 3);
            get_endeffPos_in_two_models(gc_endeff, gc_ref_endeff);
            double sumsquare_diffendeff = deepmimic_.get_endeffPosReward(n_endeff, gc_endeff, gc_ref_endeff);
            double endeff_reward = exp(-1.0 * 1.25 * sumsquare_diffendeff);
            endeffPosReward_ = endeffPosRewardCoeff_ * endeff_reward;
            endeffPosReward_raw_ = sqrt(1.25 * sumsquare_diffendeff);

            // Fourth deepmimic reward: whole body COM error by MG
            raisim::Vec<3> position;
            Eigen::VectorXd vcom1(6), vcom2(6), pel_vel(3), pel_vel_ref(3);
            get_pelvis_velocity(pel_vel, pel_vel_ref, gv_ref);
            position = anymal_->getCOM();
            vcom1 << position(0), position(1), position(2),
                    pel_vel(0), pel_vel(1), pel_vel(2);
            position = anymal_ref_->getCOM();
            vcom2 << position(0), position(1), position(2),
                    pel_vel_ref(0), pel_vel_ref(1), pel_vel_ref(2);
            double com_sdist = deepmimic_.get_comPosReward(vcom1, vcom2);
            double compos_reward = exp(-1.0 * 2.5 * com_sdist);
            comPosReward_ = comPosRewardCoeff_ * compos_reward;
            comPosReward_raw_ = sqrt(2.5 * com_sdist);

            // Fifth deepmimic reward: body root error
            // this reward function should be called at the end of all reward calculation
            // rootReward_ = rootRewardCoeff_ * deepmimic_.get_rootReward();

            // Sixth reward: root residual force
            // double sumsquare_residualForce = 0.;
            // for (int iRoot = 0; iRoot < 6; iRoot++)
            //     sumsquare_residualForce += (pTargetAction_[iRoot + nMuscles_+8]/(2*actionStd_[iRoot + nMuscles_+8]))
            // 	        *(pTargetAction_[iRoot + nMuscles_+8]/(2 * actionStd_[iRoot + nMuscles_+8]));
            // sumsquare_residualForce /= 6.;
            // double resforce_reward = exp(-1.0*2.0*sumsquare_residualForce);
            // resForceReward_ = resForceRewardCoeff_ * resforce_reward;
            // resForceReward_raw_ = sqrt(2.0*sumsquare_residualForce);

            totalReward_ = angularPosReward_ + angularVelReward_ + endeffPosReward_ + comPosReward_;

            auto time3 = std::chrono::steady_clock::now();
            time_part3_ = std::chrono::duration_cast<std::chrono::milliseconds>(time3 - time2).count();

            if (isnan(totalReward_) != 0){
                is_nan_detected_ = true;
                angularPosReward_ = 0.;
                angularVelReward_ = 0.;
                endeffPosReward_ = 0.;
                comPosReward_ = 0.;
                totalReward_ = -10.;

                std::cout << "NaN detected" << std::endl;
                std::cout << "Step Action : ";
                for (int iAction=0; iAction<actionDim_; iAction++)
                    std::cout << action(iAction) << ", ";
                std::cout << std::endl;
                std::cout << "Step Observation : ";
                for (int iObs=0; iObs<obDim_; iObs++)
                    std::cout << obDouble_(iObs) << ", ";
                std::cout << std::endl;
            }

            return (float) totalReward_;
        }

        void updateExtraInfo() final {
            // extraInfo_["test_01"] = 0.0;
            if (info_train_info_on_ == 1) {
                extraInfo_["0_angularPosReward_"]   = angularPosReward_;
                extraInfo_["0_angularVelReward_"]   = angularVelReward_;
                extraInfo_["0_endeffPosReward_"]    = endeffPosReward_;
                extraInfo_["0_comPosReward_"]       = comPosReward_;

                extraInfo_["1_angularPosRewardRaw"] = angularPosReward_raw_;
                extraInfo_["1_angularVelRewardRaw"] = angularVelReward_raw_;
                extraInfo_["1_endeffPosRewardRaw"]  = endeffPosReward_raw_;
                extraInfo_["1_comPosRewardRaw"]     = comPosReward_raw_;

                extraInfo_["2_time_part1"]          = time_part1_;
                extraInfo_["2_time_part2"]          = time_part2_;
                extraInfo_["2_time_part3"]          = time_part3_;
                // extraInfo_["3_curriculum_factor"]   = curriculumFactor_;
            }
            // if (other info flag is on) {
            // }
        }

        void updateObservation() {
            anymal_->getState(gc_, gv_);
            obDouble_.setZero(obDim_);

            /// body height
            int idx_begin = 0;
            obDouble_[idx_begin] = gc_[2]; // obDouble(0)
            idx_begin += 1;

            /// body vertical axis vector ... in world frame?
            raisim::Vec<4> quat_temp;
            raisim::Mat<3,3> rot_root = {};
            quat_temp[0] = gc_[3]; quat_temp[1] = gc_[4]; quat_temp[2] = gc_[5]; quat_temp[3] = gc_[6];
            raisim::quatToRotMat(quat_temp, rot_root);
            obDouble_.segment(idx_begin, 3) = rot_root.e().row(2); // obDouble(1:3)
            idx_begin += 3;

            /// joint angles
            set_obDouble_from_gc(obDouble_, gc_, idx_begin); // gc(7:36) --> obDouble(4:28)
            idx_begin += nJoints_;

            /// body velocities
            bodyLinearVel_ = rot_root.e().transpose() * gv_.segment(0, 3);
            bodyAngularVel_ = rot_root.e().transpose() * gv_.segment(3, 3);
            obDouble_.segment(idx_begin, 3) = bodyLinearVel_;  // obDouble(29:31)
            idx_begin += 3;
            obDouble_.segment(idx_begin, 3) = bodyAngularVel_; // obDouble(32:34)
            idx_begin += 3;

            /// joint velocities
            obDouble_.segment(idx_begin, nJoints_) = gv_.tail(nJoints_);  // obDouble(35:59)
            idx_begin += nJoints_;

            /// deepmimic phase
            // double tau = tau_random_start_ + simulationCounter_ * simulation_dt_;
            // obDouble_(idx_begin) = deepmimic_.get_phase(tau);   // obDouble(60)
            // idx_begin += 1;

            /// ground contact forces in the left and right feet
            int contact_r1_idx = 0, contact_l1_idx = 3;
            int matrowidx;

            Eigen::MatrixXd Forces(6, 3);
            Eigen::MatrixXd Positions(6, 3);
            Forces.setZero();Positions.setZero();
            std::size_t r1 = anymal_->getBodyIdx("toes_r");
            std::size_t r2 = anymal_->getBodyIdx("calcn_r");
            std::size_t l1 = anymal_->getBodyIdx("toes_l");
            std::size_t l2 = anymal_->getBodyIdx("calcn_l");
            for (auto& mycontact : anymal_->getContacts()) {
                matrowidx = -1;
                if (r1 == mycontact.getlocalBodyIndex()) {
                    matrowidx = contact_r1_idx;
                    contact_r1_idx += 1;
                }
                if (r2 == mycontact.getlocalBodyIndex()) {
                    matrowidx = 2;
                }
                if (l1 == mycontact.getlocalBodyIndex()) {
                    matrowidx = contact_l1_idx;
                    contact_l1_idx += 1;
                }
                if (l2 == mycontact.getlocalBodyIndex()) {
                    matrowidx = 5;
                }
                if (matrowidx == -1)
                    continue;

                auto force_temp = mycontact.getContactFrame().e().transpose() * mycontact.getImpulse().e() * world_->getTimeStep();
                auto position_temp = mycontact.getPosition().e().transpose();
                Forces.row(matrowidx) << force_temp[0], force_temp[1], force_temp[2];
                Positions.row(matrowidx) << position_temp[0], position_temp[1], position_temp[2];
            }

            /// yjkoo 20210422, calculate center of pressure in global frame
            raisim::Vec<3> CoP_r, CoP_l;
            raisim::Vec<3> Force_r, Force_l;

            Force_r[0] = Forces.row(0)[0] + Forces.row(1)[0] + Forces.row(2)[0];
            Force_r[1] = Forces.row(0)[1] + Forces.row(1)[1] + Forces.row(2)[1];
            Force_r[2] = Forces.row(0)[2] + Forces.row(1)[2] + Forces.row(2)[2];
            Force_l[0] = Forces.row(3)[0] + Forces.row(4)[0] + Forces.row(5)[0];
            Force_l[1] = Forces.row(3)[1] + Forces.row(4)[1] + Forces.row(5)[1];
            Force_l[2] = Forces.row(3)[2] + Forces.row(4)[2] + Forces.row(5)[2];

            CoP_r[0] = (Positions.row(0)[0] * Forces.row(0)[2] + Positions.row(1)[0] * Forces.row(1)[2]	+ Positions.row(2)[0] * Forces.row(2)[2]) / Force_r[2];
            CoP_r[1] = (Positions.row(0)[1] * Forces.row(0)[2] + Positions.row(1)[1] * Forces.row(1)[2]	+ Positions.row(2)[1] * Forces.row(2)[2]) / Force_r[2];
            CoP_r[2] = 0.;

            CoP_l[0] = (Positions.row(3)[0] * Forces.row(3)[2] + Positions.row(4)[0] * Forces.row(4)[2]	+ Positions.row(5)[0] * Forces.row(5)[2]) / Force_l[2];
            CoP_l[1] = (Positions.row(3)[1] * Forces.row(3)[2] + Positions.row(4)[1] * Forces.row(4)[2]	+ Positions.row(5)[1] * Forces.row(5)[2]) / Force_l[2];
            CoP_l[2] = 0.;

            /// yjkoo 20210422, calculate center of pressure in pelvis frame
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

            raisim::Vec<3> trs_zero = {0., 0., 0.};
            auto Force_r_local = pt_coord_txf(pelvis_R_inverse, trs_zero, Force_r);
            auto Force_l_local = pt_coord_txf(pelvis_R_inverse, trs_zero, Force_l);
            auto CoP_r_local = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, CoP_r);
            auto CoP_l_local = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, CoP_l);

            if ((abs(Force_r[2]) < 0.005) || (isnan(CoP_r[0]) !=0 )) {
                std::size_t idx_rankle = anymal_->getFrameIdxByName("ankle_r");
                raisim::Vec<3> pos_temp;
                anymal_->getFramePosition(idx_rankle, pos_temp);
                pos_temp = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, pos_temp);
                CoP_r_local = pos_temp;
                Force_r_local.setZero();
                //std::cout << "right swing: " << CoP_r[0] << std::endl;
            }

            if ((abs(Force_l[2]) < 0.005) || (isnan(CoP_l[0]) != 0)) {
                std::size_t idx_rankle = anymal_->getFrameIdxByName("ankle_l");
                raisim::Vec<3> pos_temp;
                anymal_->getFramePosition(idx_rankle, pos_temp);
                pos_temp = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, pos_temp);
                CoP_l_local = pos_temp;
                Force_l_local.setZero();
                //std::cout << "left swing" << std::endl;
            }

            obDouble_.segment(idx_begin, 3) << Force_r_local[0], Force_r_local[1], Force_r_local[2]; // obDouble(61:63)
            idx_begin += 3;
            obDouble_.segment(idx_begin, 3) << CoP_r_local[0], CoP_r_local[1], CoP_r_local[2]; // obDouble(64:66)
            idx_begin += 3;
            obDouble_.segment(idx_begin, 3) << Force_l_local[0], Force_l_local[1], Force_l_local[2]; // obDouble(67:69)
            idx_begin += 3;
            obDouble_.segment(idx_begin, 3) << CoP_l_local[0], CoP_l_local[1], CoP_l_local[2]; // obDouble(70:72)
            idx_begin += 3;

            /// observations of AMP targets
            obDouble_(idx_begin) = target_speed_;
            idx_begin += 1;
            obDouble_.segment(idx_begin, 3) << target_direction_[0], target_direction_[1], target_direction_[2];
            idx_begin += 3;

            /// CoP and GRF visualization
            Right_contact_->points.clear();
            Right_contact_->points.push_back({ CoP_r[0], CoP_r[1], CoP_r[2] });
            Right_contact_->points.push_back({ CoP_r[0] + Force_r[0], CoP_r[1] + Force_r[1], CoP_r[2] + Force_r[2] });

            Left_contact_->points.clear();
            Left_contact_->points.push_back({ CoP_l[0], CoP_l[1], CoP_l[2] });
            Left_contact_->points.push_back({ CoP_l[0] + Force_l[0], CoP_l[1] + Force_l[1], CoP_l[2] + Force_l[2] });

            /// muscle state (tendon length(=muscle force) and activation)
//          for (int iter = 0; iter < nMuscles_; iter ++)
//              obDouble_[idx_begin+iter] = Muscles[iter].TendonLength;
//          idx_begin += 92;
//          for (int iter = 0; iter < nMuscles_; iter ++)
//              obDouble_[idx_begin+iter] = Muscles[iter].Activation;
//          idx_begin += 92;
//          for (int iter = 0; iter < nMuscles_; iter ++)
//              obDouble_[idx_begin+iter] = Muscles[iter].m_muscleLength;
//          idx_begin += 92;

            // It should be added ... YJKOO
//          for (int iter = 0; iter < nmuscles; iter ++) {
//              if ( isnan(Muscles[iter].Length) || isnan(Muscles[iter].OptFiberLength) || isnan(Muscles[iter].TendonSlackLength) ){
//                  obDouble_[45+3*iter] = Muscles[iter].OptFiberLength;    // fiber length
//                  obDouble_[45+3*iter+1] = 0;      // fiber velocity
//                  obDouble_[45+3*iter+2] = Muscles[iter].TendonSlackLength;    // tendon length
//              }
//              else{
//                  obDouble_[45+3*iter] = Muscles[iter].FiberLength;       // fiber length
//                  obDouble_[45+3*iter+1] = Muscles[iter].FiberVelocity;   // fiber velocity
//                  obDouble_[45+3*iter+2] = Muscles[iter].TendonLength;    // tendon length
//              }
//          }
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            /// convert it to float
            ob = obDouble_.cast<float>();
        }

        void observe_amp(Eigen::Ref<EigenVec> ob) final {
        }

        bool isTerminalState(float &terminalReward) final {
            terminalReward = float(terminalRewardCoeff_);

            std::size_t right_femur = anymal_->getBodyIdx("femur_r");
            std::size_t right_tibia = anymal_->getBodyIdx("tibia_r");
            std::size_t left_femur = anymal_->getBodyIdx("femur_l");
            std::size_t left_tibia = anymal_->getBodyIdx("tibia_l");

            std::vector<Contact> contacts = anymal_->getContacts();
            for(auto& contact: anymal_->getContacts()) {
                /// if the ground contact body is not feet
                if(contact.getPairObjectIndex() == ground_->getIndexInWorld())
                    if (footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end())
                        return true;

                /// when the two legs are crossed... for early termination
                if(contact.isSelfCollision()) {
                    if(contacts[contact.getPairContactIndexInPairObject()].getlocalBodyIndex() == right_femur  && contact.getlocalBodyIndex() == left_femur)
                        return true;
                    else if (contacts[contact.getPairContactIndexInPairObject()].getlocalBodyIndex() == left_femur  && contact.getlocalBodyIndex() == right_femur)
                        return true;
                    if(contacts[contact.getPairContactIndexInPairObject()].getlocalBodyIndex() == right_tibia  && contact.getlocalBodyIndex() == left_tibia)
                        return true;
                    else if (contacts[contact.getPairContactIndexInPairObject()].getlocalBodyIndex() == left_tibia  && contact.getlocalBodyIndex() == right_tibia)
                        return true;
                }
            }

            if (is_nan_detected_)
                return true;

            terminalReward = 0.f;
            return false;
        }


        void get_pelvis_velocity(Eigen::VectorXd &pelvis_vel, Eigen::VectorXd &pelvis_vel_ref, Eigen::VectorXd &gv_ref) {
            raisim::Mat<3, 3> pelvis_R{};
            anymal_->getBaseOrientation(pelvis_R);
            raisim::Mat<3, 3> pelvis_R_inverse = pelvis_R.transpose();

            raisim::Mat<3, 3> pelvis_R_ref{};
            anymal_ref_->getBaseOrientation(pelvis_R_ref);
            raisim::Mat<3, 3> pelvis_R_ref_inverse = pelvis_R_ref.transpose();

            raisim::Vec<3> pel_vel, pel_vel_local, pel_vel_ref, pel_vel_ref_local;
            raisim::Vec<3> trs_zero = {0., 0., 0.};

            pel_vel = { gv_(0), gv_(1), gv_(2) };
            pel_vel_local = pt_coord_txf(pelvis_R_inverse, trs_zero, pel_vel);

            pel_vel_ref = { gv_ref(0), gv_ref(1), gv_ref(2) };
            pel_vel_ref_local = pt_coord_txf(pelvis_R_inverse, trs_zero, pel_vel_ref);

            pelvis_vel.segment(0, 3) << pel_vel_local(0), pel_vel_local(1), pel_vel_local(2);
            pelvis_vel_ref.segment(0, 3) << pel_vel_ref_local(0), pel_vel_ref_local(1), pel_vel_ref_local(2);
        }


        void get_endeffPos_in_two_models(Eigen::MatrixXd &gc_endeff, Eigen::MatrixXd &gc_ref_endeff) {
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

            raisim::Vec<3> pelvis_T_ref;
            raisim::Mat<3, 3> pelvis_R_ref{};
            anymal_ref_->getBasePosition(pelvis_T_ref);
            anymal_ref_->getBaseOrientation(pelvis_R_ref);
            raisim::Mat<3, 3> pelvis_R_ref_inverse = pelvis_R_ref.transpose();
            raisim::Vec<3> pelvis_T_ref_inverse = {
                    -pelvis_R_ref_inverse[0] * pelvis_T_ref[0] - pelvis_R_ref_inverse[3] * pelvis_T_ref[1] - pelvis_R_ref_inverse[6] * pelvis_T_ref[2],
                    -pelvis_R_ref_inverse[1] * pelvis_T_ref[0] - pelvis_R_ref_inverse[4] * pelvis_T_ref[1] - pelvis_R_ref_inverse[7] * pelvis_T_ref[2],
                    -pelvis_R_ref_inverse[2] * pelvis_T_ref[0] - pelvis_R_ref_inverse[5] * pelvis_T_ref[1] - pelvis_R_ref_inverse[8] * pelvis_T_ref[2]
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

            /// positions for r_wrist, l_wrist, r_ankle, l_ankle in the reference model
            // anymal_ref_->getFramePosition(idx_neck, position);
            // gc_ref_endeff.row(0) << position(0), position(1), position(2);

            anymal_ref_->getFramePosition(idx_rwrist, position);
            position = pt_coord_txf(pelvis_R_ref_inverse, pelvis_T_ref_inverse, position);
            gc_ref_endeff.row(0) << position(0), position(1), position(2);

            anymal_ref_->getFramePosition(idx_lwrist, position);
            position = pt_coord_txf(pelvis_R_ref_inverse, pelvis_T_ref_inverse, position);
            gc_ref_endeff.row(1) << position(0), position(1), position(2);

            anymal_ref_->getFramePosition(idx_rankle, position);
            position = pt_coord_txf(pelvis_R_ref_inverse, pelvis_T_ref_inverse, position);
            gc_ref_endeff.row(2) << position(0), position(1), position(2);

            anymal_ref_->getFramePosition(idx_lankle, position);
            position = pt_coord_txf(pelvis_R_ref_inverse, pelvis_T_ref_inverse, position);
            gc_ref_endeff.row(3) << position(0), position(1), position(2);

            /// position for top_of_head
            raisim::Vec<3> pt_top_of_head; // top of head in torso coord
            pt_top_of_head[0] = 0.00105107; pt_top_of_head[1] = 0.663102; pt_top_of_head[2] = 0.;

            raisim::Mat<3, 3> torso_R{};
            raisim::Vec<3> torso_T;
            anymal_->getLink("torso").getPose(torso_T, torso_R);
            position = pt_coord_txf(torso_R, torso_T, pt_top_of_head);
            position = pt_coord_txf(pelvis_R_inverse, pelvis_T_inverse, position);
            gc_endeff.row(4) << position(0), position(1), position(2);

            raisim::Mat<3, 3> torso_R_ref{};
            raisim::Vec<3> torso_T_ref;
            anymal_ref_->getLink("torso").getPose(torso_T_ref, torso_R_ref);
            position = pt_coord_txf(torso_R_ref, torso_T_ref, pt_top_of_head);
            position = pt_coord_txf(pelvis_R_ref_inverse, pelvis_T_ref_inverse, position);
            gc_ref_endeff.row(4) << position(0), position(1), position(2);
        }

        static raisim::Vec<3> pt_coord_txf(raisim::Mat<3,3> &rot, raisim::Vec<3> &trs, raisim::Vec<3> &pt) {
            raisim::Vec<3> pt_txfed = {
                    rot[0] * pt[0] + rot[3] * pt[1] + rot[6] * pt[2] + trs[0],
                    rot[1] * pt[0] + rot[4] * pt[1] + rot[7] * pt[2] + trs[1],
                    rot[2] * pt[0] + rot[5] * pt[1] + rot[8] * pt[2] + trs[2]
            };

            return pt_txfed;
        }

        void update_seg_txf() {
            raisim::Vec<3> TrsVec;
            raisim::Mat<3, 3> RotMat{};

            /// rotation and translation of pelvis
            anymal_->getBasePosition(TrsVec);
            anymal_->getBaseOrientation(RotMat);
            muscle_trs_[0] = TrsVec;
            muscle_rot_[0] = RotMat;

            /// rotation and translation of other segments
            auto str_temp = anymal_->getBodyNames();
            for (int i = 1; i < anymal_->getNumberOfJoints(); i++) {
                anymal_->getLink(str_temp.at(i)).getPose(TrsVec, RotMat);
                muscle_trs_[i] = TrsVec;
                muscle_rot_[i] = RotMat;
            }
        }

        void ReadMuscleInitialLength(std::string FileName, double *L0) {
            std::ifstream jsonfile(FileName);
            nlohmann::json jsondata;
            jsonfile >> jsondata;
            std::string strtemp;

            for (int iter = 0; iter < 92; iter++) {    // setting muscle parameters
                L0[iter] = jsondata[Muscles_[iter].m_strMuscleName][0];
            }
        }

        void curriculumUpdate() final {
            if (curriculumFactor_ < 1.0 && curriculumPhase_ == 1)
                curriculumFactor_ += 0.025;
            else
                curriculumFactor_ = 1.0;
        }


        // prototypes of public functions
        void set_obDouble_from_gc(Eigen::VectorXd &obDouble, Eigen::VectorXd &gc, int idx_begin);
        int set_gc_from_action(Eigen::VectorXd &targetgc, Eigen::VectorXd &action);
        void set_joint_pos_action_scales(Eigen::VectorXd &actionMean, Eigen::VectorXd &actionStd);
        void set_joint_torque_action_scales(Eigen::VectorXd &actionMean, Eigen::VectorXd &actionStd);
        void set_jointPgain_jointDgain(Eigen::VectorXd &jointPgain, Eigen::VectorXd &jointDgain);
        void GetJointInfo();
        void GetScaleFactor(std::string file_path);
        void ReadMuscleModel(const std::string& FileName);

    private:
        std::size_t gcDim_, gvDim_, nJoints_;
        bool visualizable_ = false;
        raisim::ArticulatedSystem *anymal_;
        raisim::Ground *ground_;
        Eigen::VectorXd gc_, gv_;
        Eigen::VectorXd PdTarget_gc_, pTargetAction_, PdTarget_gv_;
        double terminalRewardCoeff_ = -10.;
        double angularPosRewardCoeff_ = 0., angularPosReward_ = 0., angularPosReward_raw_ = 0.;
        double angularVelRewardCoeff_ = 0., angularVelReward_ = 0., angularVelReward_raw_ = 0.;
        double endeffPosRewardCoeff_ = 0., endeffPosReward_ = 0., endeffPosReward_raw_ = 0.;
        double comPosRewardCoeff_ = 0., comPosReward_ = 0., comPosReward_raw_ = 0.;

        double totalReward_ = 0.;
        std::size_t simulationCounter_ = 0;

        Eigen::VectorXd actionMean_, actionStd_;
        Eigen::VectorXd jointPosActionMean_, jointPosActionStd_;
        Eigen::VectorXd jointTorqueActionMean_, jointTorqueActionStd_;
        Eigen::VectorXd obDouble_;
        Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
        std::set<size_t> footIndices_;

        /// AMP variables
        double target_speed_;
        Eigen::Vector3d target_direction_;

        /// Debugging variables
        bool is_nan_detected_ = false;
        double time_part1_ = 0., time_part2_ = 0., time_part3_ = 0.;

        /// Curriculum variables
        std::size_t curriculumPhase_ = 0;
        double curriculumFactor_ = 0.0;

        /// Deepmimic variables
        DeepmimicUtility deepmimic_;
        raisim::ArticulatedSystem *anymal_ref_;
        raisim::ArticulatedSystemVisual *anymal_vis_;
        raisim::World world_ref_;
        std::size_t num_ref_motion_;
        double tau_random_start_;
        double gc_ref_body_y_offset_ = 0.0; // SKOO 20200712 lateral offset of the model
        std::vector<JointInfo> joint_info_;

        /// Muscle variables
        std::size_t nMuscles_;
        std::vector<MuscleModel> Muscles_;
        std::vector<raisim::Vec<3>> muscle_trs_;
        std::vector<raisim::Mat<3,3>> muscle_rot_;
        std::vector<std::string> scale_link_;
        std::vector<double> scale_value_;
        std::vector<double> JointTorque_muscle_; // nJoints_
        std::vector<double> JointTorque_direct_; // nJoints_
        std::vector<double> JointTorque_pdcont_; // nJoints_

        raisim::PolyLine *Right_contact_;
        raisim::PolyLine *Left_contact_;

        std::size_t info_train_info_on_ = 0;
    };


    int ENVIRONMENT::set_gc_from_action(Eigen::VectorXd &targetgc, Eigen::VectorXd &action)
    {
        std::string joint_type;
        int ngv;
        int gc_index[4];
        int count_gc = 0;
        int idx_begin = 0;

        for (int ijoint=2; ijoint<joint_info_.size(); ijoint++) {   // exclude root trs and rot
            joint_info_[ijoint].get_gc_index(gc_index);
            joint_type = joint_info_[ijoint].get_joint_type();
            ngv = joint_info_[ijoint].get_ngv();

            if (joint_type == "rot3") {
                double v1[3], q1[4];
                v1[0] = action(idx_begin+0);
                v1[1] = action(idx_begin+1);
                v1[2] = action(idx_begin+2);
                Quaternion::rotvec2quat(q1, v1);
                targetgc(gc_index[0]) = q1[0];
                targetgc(gc_index[1]) = q1[1];
                targetgc(gc_index[2]) = q1[2];
                targetgc(gc_index[3]) = q1[3];
                count_gc += 4;
            }
            if (joint_type == "rot1") {
                targetgc(gc_index[0]) = action(idx_begin);
                count_gc += 1;
            }
            if (joint_type == "trs3") {
                targetgc(gc_index[0]) = action(idx_begin+0);
                targetgc(gc_index[1]) = action(idx_begin+1);
                targetgc(gc_index[2]) = action(idx_begin+2);
                count_gc += 3;
            }

            idx_begin += ngv;
        }
        return count_gc;
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


    void ENVIRONMENT::set_joint_pos_action_scales(Eigen::VectorXd &actionMean, Eigen::VectorXd &actionStd)
    {
        int ngv;
        double action_mean[3];
        double action_std[3];
        int idx_begin = 0;

        for (int ijoint=2; ijoint<joint_info_.size(); ijoint++) {   // exclude root trs and rot
            ngv = joint_info_[ijoint].get_ngv();
            joint_info_[ijoint].get_pos_action_mean(action_mean);
            joint_info_[ijoint].get_pos_action_std(action_std);

            for (int igv=0; igv<ngv; igv++) {
                actionMean(idx_begin+igv) = action_mean[igv];  // in the order of joint
                actionStd(idx_begin+igv)  = action_std[igv];   // in the order of joint
            }
            idx_begin += ngv;
        }
    }


    void ENVIRONMENT::set_joint_torque_action_scales(Eigen::VectorXd &actionMean, Eigen::VectorXd &actionStd)
    {
        int ngv;
        double action_mean[3];
        double action_std[3];
        int idx_begin = 0;

        for (int ijoint=2; ijoint<joint_info_.size(); ijoint++) {   // exclude root trs and rot
            ngv = joint_info_[ijoint].get_ngv();
            joint_info_[ijoint].get_torque_action_mean(action_mean);
            joint_info_[ijoint].get_torque_action_std(action_std);

            for (int igv=0; igv<ngv; igv++) {
                actionMean(idx_begin+igv) = action_mean[igv];   // in the order of joint
                actionStd(idx_begin+igv)  = action_std[igv];    // in the order of joint
            }
            idx_begin += ngv;
        }
    }


    void ENVIRONMENT::set_jointPgain_jointDgain(Eigen::VectorXd &jointPgain, Eigen::VectorXd &jointDgain)
    {
        int ngv;
        double joint_pgain[3], joint_dgain[3];
        int idx_begin = 0;

        for (int ijoint=0; ijoint<joint_info_.size(); ijoint++) {    // including the two root joints
            ngv = joint_info_[ijoint].get_ngv();
            joint_info_[ijoint].get_pgain(joint_pgain);
            joint_info_[ijoint].get_dgain(joint_dgain);

            for (int igv=0; igv<ngv; igv++) {
                jointPgain(idx_begin+igv) = joint_pgain[igv];      // in the order of joint
                jointDgain(idx_begin+igv) = joint_dgain[igv];      // in the order of joint
            }
            idx_begin += ngv;
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


    void ENVIRONMENT::GetScaleFactor(std::string file_path) {
        std::ifstream f(file_path);
        std::stringstream ss;
        ss << f.rdbuf();
        std::string urdf_str = ss.str();
        //auto urdf_str_size = urdf_str.size();

        scale_link_[0] = "pelvis";
        scale_link_[1] = "femur_r";
        scale_link_[2] = "tibia_r";
        scale_link_[3] = "talus_r";
        scale_link_[4] = "calcn_r";
        scale_link_[5] = "toes_r";
        scale_link_[6] = "femur_l";
        scale_link_[7] = "tibia_l";
        scale_link_[8] = "talus_l";
        scale_link_[9] = "calcn_l";
        scale_link_[10] = "toes_l";
        scale_link_[11] = "torso";
        scale_link_[12] = "humerus_r";
        scale_link_[13] = "ulna_r";
        scale_link_[14] = "radius_r";
        scale_link_[15] = "hand_r";
        scale_link_[16] = "humerus_l";
        scale_link_[17] = "ulna_l";
        scale_link_[18] = "radius_l";
        scale_link_[19] = "hand_l";

        std::size_t idx_link[20], idx_scale1[20], idx_scale2[20];

        for (int iter = 0; iter < 20; iter++) {
            idx_link[iter] = urdf_str.find(scale_link_[iter]);
            /// print link name
            //std::cout << temp.substr(idx_link[iter], scale_link_[iter].length()) << std::endl;

            idx_scale1[iter] = urdf_str.find("scale=", idx_link[iter]) + 7;
            idx_scale2[iter] = urdf_str.find(" ", idx_scale1[iter]);
            std::size_t scale_length = (idx_scale2[iter]) - (idx_scale1[iter]);
            //std::cout << "scale: " << temp.substr(idx_scale1[iter], scale_length) << std::endl;

            scale_value_[iter] = std::stod(urdf_str.substr(idx_scale1[iter], scale_length));
            //std::cout << "scale: " << scale_value_[iter] << std::endl;
        }
    }


    void ENVIRONMENT::ReadMuscleModel(const std::string& FileName) {
        std::ifstream jsonfile(FileName);
        if (!jsonfile) std::cout << "Failed to open the muscle definitionfile" << std::endl;

        nlohmann::json jsondata;
        jsonfile >> jsondata;
        std::string strtemp;
        int nmuscles_temp = jsondata["nMuscles"];    // the number of muscles
        assert(nmuscles_temp == 92);

        for (int iter = 0; iter < nmuscles_temp; iter++) {    // setting muscle parameters
            Muscles_[iter].m_strMuscleName = jsondata["Muscles"][iter]["Name"];
            // std::cout << Muscles[iter].m_strMuscleName << std::endl;

            strtemp = jsondata["Muscles"][iter]["max_isometric_force"];
            Muscles_[iter].Fiso = std::stod(strtemp);

            strtemp = jsondata["Muscles"][iter]["optimal_fiber_length"];
            Muscles_[iter].OptFiberLength = std::stod(strtemp);

            strtemp = jsondata["Muscles"][iter]["tendon_slack_length"];
            Muscles_[iter].TendonSlackLength = std::stod(strtemp);

            strtemp = jsondata["Muscles"][iter]["pennation_angle"];
            Muscles_[iter].PennationAngle = std::stod(strtemp);

            Muscles_[iter].m_nTargetJoint = jsondata["Muscles"][iter]["nTargetJoint"];
            if (Muscles_[iter].m_nTargetJoint == 1) {   // one joint muscle
                Muscles_[iter].m_strTargetJoint[0] = jsondata["Muscles"][iter]["TargetJoint"]["Joint"];
                Muscles_[iter].m_nTargetPath[0][0] = jsondata["Muscles"][iter]["TargetJoint"]["Path"][0];
                Muscles_[iter].m_nTargetPath[0][1] = jsondata["Muscles"][iter]["TargetJoint"]["Path"][1];
            }
            else {
                for (int iTargetJoint = 0; iTargetJoint < Muscles_[iter].m_nTargetJoint; iTargetJoint++) {
                    Muscles_[iter].m_strTargetJoint[iTargetJoint] = jsondata["Muscles"][iter]["TargetJoint"][iTargetJoint]["Joint"];
                    Muscles_[iter].m_nTargetPath[iTargetJoint][0] = jsondata["Muscles"][iter]["TargetJoint"][iTargetJoint]["Path"][0];
                    Muscles_[iter].m_nTargetPath[iTargetJoint][1] = jsondata["Muscles"][iter]["TargetJoint"][iTargetJoint]["Path"][1];
                }
            }
            int nPathPoint = jsondata["Muscles"][iter]["nPath"];
            Muscles_[iter].m_nPathPoint = nPathPoint; assert(nPathPoint < 9);

            double temp_double[3];
            std::string mystring;
            for (int ipath = 0; ipath < nPathPoint; ipath++) {
                std::stringstream ss2;
                ss2 << ipath + 1;
                mystring = "path" + ss2.str();

                temp_double[0] = jsondata["Muscles"][iter]["PathSet"][mystring]["locationx"];
                temp_double[1] = jsondata["Muscles"][iter]["PathSet"][mystring]["locationy"];
                temp_double[2] = jsondata["Muscles"][iter]["PathSet"][mystring]["locationz"];

                Muscles_[iter].PathPos[ipath] = { temp_double[0], temp_double[1], temp_double[2] };
                Muscles_[iter].PathLink[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["body"];
                Muscles_[iter].PathType[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["type"];
                if (Muscles_[iter].PathType[ipath] == "Conditional") {
                    Muscles_[iter].Coordinate[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["coordinate"];
                    Muscles_[iter].Range[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["range"];
                }
                if (Muscles_[iter].PathType[ipath] == "Moving") {
                    Muscles_[iter].CoordinateX[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["x_coordinate"];
                    Muscles_[iter].CoordinateY[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["y_coordinate"];
                    Muscles_[iter].CoordinateZ[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["z_coordinate"];
                    Muscles_[iter].x_x[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["x_location"]["x"];
                    Muscles_[iter].x_y[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["x_location"]["y"];
                    Muscles_[iter].y_x[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["y_location"]["x"];
                    Muscles_[iter].y_y[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["y_location"]["y"];
                    Muscles_[iter].z_x[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["z_location"]["x"];
                    Muscles_[iter].z_y[ipath] = jsondata["Muscles"][iter]["PathSet"][mystring]["z_location"]["y"];
                    double d = 0.0;
                    std::stringstream sx(Muscles_[iter].x_x[ipath]);
                    while (sx >> d)
                        Muscles_[iter].vx.push_back(d);
                    std::stringstream sy(Muscles_[iter].x_y[ipath]);
                    while (sy >> d)
                        Muscles_[iter].vy.push_back(d);

                    std::stringstream sx2(Muscles_[iter].y_x[ipath]);
                    while (sx2 >> d)
                        Muscles_[iter].vx2.push_back(d);
                    std::stringstream sy2(Muscles_[iter].y_y[ipath]);
                    while (sy2 >> d)
                        Muscles_[iter].vy2.push_back(d);

                    std::vector<double> b_temp, c_temp, d_temp;
                    b_temp.clear();c_temp.clear();d_temp.clear();
                    MuscleModel::calcCoefficients(Muscles_[iter].vx, Muscles_[iter].vy, b_temp, c_temp, d_temp);
                    for (int j = 0; j < Muscles_[iter].vx.size(); j++) {
                        Muscles_[iter].x_b.push_back(b_temp.at(j));
                        Muscles_[iter].x_c.push_back(c_temp.at(j));
                        Muscles_[iter].x_d.push_back(d_temp.at(j));
                    }
                    b_temp.clear();c_temp.clear();d_temp.clear();
                    MuscleModel::calcCoefficients(Muscles_[iter].vx2, Muscles_[iter].vy2, b_temp, c_temp, d_temp);
                    for (int j = 0; j < Muscles_[iter].vx2.size(); j++) {
                        Muscles_[iter].y_b.push_back(b_temp.at(j));
                        Muscles_[iter].y_c.push_back(c_temp.at(j));
                        Muscles_[iter].y_d.push_back(d_temp.at(j));
                    }
                }
            } // for ipath
        }
    }
}

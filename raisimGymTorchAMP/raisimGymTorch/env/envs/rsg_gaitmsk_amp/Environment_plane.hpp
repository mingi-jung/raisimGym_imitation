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

            /// Reward coefficients
            READ_YAML(double, angularPosRewardCoeff_, cfg["angularPosRewardCoeff"])
            READ_YAML(double, angularVelRewardCoeff_, cfg["angularVelRewardCoeff"])
            READ_YAML(double, endeffPosRewardCoeff_, cfg["endeffPosRewardCoeff"])
            READ_YAML(double, comPosRewardCoeff_, cfg["comPosRewardCoeff"])

            /// Step info flags
            READ_YAML(int, info_train_info_on_, cfg["train_info_on"])

            /// curriculum learning
            // READ_YAML(int, curriculumPhase_, cfg["curriculum_phase"])
            // curriculumFactor_ = 0.0;

            /// actuation mode
            READ_YAML(int, use_raisim_pdcontrol_on_, cfg["use_rasim_pdcontrol"]);
            READ_YAML(int, use_muscle_on_, cfg["use_muscle"]);
            READ_YAML(int, use_direct_torque_on_, cfg["use_direct_torque"]);

            /// add objects
            std::string URDF_filename = resourceDir_ + "/urdf/MSK_GAIT2392_model_subject06.urdf";
            anymal_ = world_->addArticulatedSystem(URDF_filename);
            anymal_->setName("anymal");

            /// load urdf-specific joint information
            GetJointInfo();

            if (use_raisim_pdcontrol_on_)
                anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
            else
                anymal_->setControlMode(raisim::ControlMode::FORCE_AND_TORQUE);

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
            if (use_raisim_pdcontrol_on_) {
                Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
                set_jointPgain_jointDgain(jointPgain, jointDgain);
                anymal_->setPdGains(jointPgain, jointDgain);
            }

            /// initialize feedforward forces
            anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

            /// indices of links that should not make contact with ground
            footIndices_.insert(anymal_->getBodyIdx("toes_r"));
            footIndices_.insert(anymal_->getBodyIdx("calcn_r"));
            footIndices_.insert(anymal_->getBodyIdx("toes_l"));
            footIndices_.insert(anymal_->getBodyIdx("calcn_l"));

            /// action scaling parameters - muscle, pose, torque
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

            double spd_temp = 0.0;
            for (int irefmotion=0; irefmotion<num_ref_motion_; irefmotion++) {
                READ_YAML(double, spd_temp, cfg["ref_motion_speed_"+std::to_string(irefmotion+1)])
                ref_motion_speed_.push_back(spd_temp);
            }

            for (int irefmotion=0; irefmotion<num_ref_motion_; irefmotion++) {
                // deepmimic_.set_current_ref_motion_index(irefmotion);
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
                Muscles_[imuscles].SetControlDt(control_dt_);
                Muscles_[imuscles].initialize(muscle_trs_, muscle_rot_);
                Muscles_[imuscles].SetHillTypeOn();
                Muscles_[imuscles].SetScaleMuscle(scale_link_, scale_value_, muscle_trs_, muscle_rot_, muscle_initial_length[imuscles]);
                Muscles_[imuscles].SetFiso(2.0*Muscles_[imuscles].GetFiso());  // increase the maximum muscle strength by factor of two
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
            std::size_t idx_refmotion = deepmimic_.get_idx_refmotion();
            double gait_speed = ref_motion_speed_[idx_refmotion];

            /// Random start of simulation
            tau_random_start_ = deepmimic_.get_tau_random_start();

            for (int iMuscle = 0; iMuscle < nMuscles_; iMuscle++) {
                Muscles_[iMuscle].SetActivation(-0.01); // See Muscle.hpp Line# 745
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

            /// AMP goals
            // std::random_device rd;  // random device
            // std::mt19937 mersenne(rd()); // random generator, a mersenne twister
            // std::uniform_real_distribution<double> distribution(0.5, 5.0);
            // double random_number = distribution(mersenne);
            // target_speed_ = random_number;
            target_speed_ = gait_speed; // for this specific refmotion data
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

            /// joint actuation by pd control
            PdTarget_gc_.setZero(gcDim_); PdTarget_gv_.setZero(gvDim_);

            Eigen::VectorXd pTargetAction_temp = pTargetAction_.segment(nMuscles_, nJoints_);          // 92 - 116 (25)
            int num_gc_temp = set_gc_from_action(PdTarget_gc_, pTargetAction_temp);  // action(25) into gc(7+30)

            if (use_raisim_pdcontrol_on_) {
                anymal_->setPdTarget(PdTarget_gc_, PdTarget_gv_);
            } else {
                std::vector<double> JointTorque_muscle(nJoints_, 0.);
                std::vector<double> JointTorque_direct(nJoints_, 0.);
                std::vector<double> JointTorque_pdcont(nJoints_, 0.);

                // We replace setPdTarget with our own stable PD controller
                anymal_->getState(gc_, gv_);

                /// joint actuation by pd control; see stable PD control by Tan el al., 2011
                std::string joint_type;
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
                    // kd[0] = 0.0; kd[1] = 0.0; kd[2] = 0.0;

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

                        // stable pd controller in Tan et al., 2011
                        // - kp * (-1*ln(q1.inv * qbar) + control_dt_*qdot1)
                        // - kd * (qdot1 + control_dt_*(qdot2-qdot1)/dt_ - qdotbar)

                        for (int ii=0; ii<3; ii++) {
                            double p_temp = (-2 * vq[ii] + control_dt_ * gv_[gv_index[ii]]);
                            // double d_temp = (gv_[gv_index[ii]] + (gv_[gv_index[ii]] - gv_pre_[gv_index[ii]]) - PdTarget_gv_[gv_index[ii]]);
                            // JointTorque_pdcont_[gv_index[ii] - 6] = - kp[ii] * p_temp - kd[ii] * d_temp;
                            // 20210401 SKOO I could not make the Kd-term work
                            JointTorque_pdcont[gv_index[ii] - 6] = - kp[ii] * p_temp;
                        }
                    }
                    if (joint_type == "rot1") {
                        JointTorque_pdcont[gv_index[0]-6] =
                                - kp[0]*(gc_[gc_index[0]] + control_dt_*gv_[gv_index[0]] - PdTarget_gc_[gc_index[0]]);
                                // - kd[0]*(gv_[gv_index[0]] + (gv_[gv_index[0]] - gv_pre_[gv_index[0]]) - PdTarget_gv_[gv_index[0]]);
                    }
                    if (joint_type == "trs3") {
                        for (int ii=0; ii<3; ii++)
                            JointTorque_pdcont[gv_index[ii]-6] =
                                - kp[ii]*(gc_[gc_index[ii]] + control_dt_*gv_[gv_index[ii]] - PdTarget_gc_[gc_index[ii]]);
                                // - kd[ii]*(gv_[gv_index[ii]] + (gv_[gv_index[ii]] - gv_pre_[gv_index[ii]]) - PdTarget_gv_[gv_index[ii]]);
                    }
                }
                // gv_pre_ = gv_; // It is used to calculate the kd-term

                if (use_muscle_on_) {
                    // when we use the muscles in the lower limb
                    // we use the pd control actuation only for the shoulder and elbow joints
                    for (int iJoint = 0; iJoint < nJoints_; iJoint++) {
                        if (iJoint < 3 || iJoint > 10)
                            JointTorque_pdcont[iJoint] = 0.0;
                    }
                }

                /// joint actuation by muscles
                if (use_muscle_on_) {
                    update_seg_txf();
                    std::vector<double> JointTorque_muscle_temp(31, 0.);
                    for (int iMuscle = 0; iMuscle < nMuscles_; iMuscle++) {  // 0 - 91
                        Muscles_[iMuscle].SetExcitation(pTargetAction_[iMuscle]);
                        Muscles_[iMuscle].Update(muscle_trs_, muscle_rot_, JointTorque_muscle_temp); // resultant joint torques
                    }
                    for (int iJoint = 0; iJoint < nJoints_; iJoint++)
                        JointTorque_muscle[iJoint] = JointTorque_muscle_temp[6 + iJoint];
                }

                /// joint actuation direct
                if (use_direct_torque_on_) {
                    for (int iJoint = 0; iJoint < nJoints_; iJoint++)
                        JointTorque_direct[iJoint] = pTargetAction_[nMuscles_ + nJoints_ + iJoint];
                }

                /// application of the actuation torques to joint actuators
                raisim::VecDyn JointTorque = anymal_->getFeedForwardGeneralizedForce(); // gvDim_
                for (int iJoint = 0; iJoint < 6; iJoint++)
                    JointTorque[iJoint] = 0.0;  // input zeros for base body actuations

                for (int iJoint = 0; iJoint < nJoints_; iJoint++) {
                    JointTorque[6 + iJoint] = JointTorque_pdcont[iJoint]
                            + JointTorque_muscle[iJoint] + JointTorque_direct[iJoint];
                }

                anymal_->setGeneralizedForce(JointTorque);
            }

            /// timers to measure the time for reward calculation
            auto time1 = std::chrono::steady_clock::now();
            time_part1_ = std::chrono::duration_cast<std::chrono::milliseconds>(time1 - time0).count();

            /// DeepMimic; gc and gv initialization
            Eigen::VectorXd gc_ref, gv_ref;
            gc_ref.setZero(gcDim_);
            gv_ref.setZero(gvDim_);


            /// visualization in the unity
            int n_integrate = int(control_dt_ / simulation_dt_ + 1e-10);
            for (int i=0; i<n_integrate; i++) {
                if(server_) server_->lockVisualizationServerMutex();
                world_->integrate();    // for simulation_dt_ ... 0.001 sec
                simulationCounter_++;   // We need it to animate the reference motion

                if(server_) server_->unlockVisualizationServerMutex();
            }


            auto time2 = std::chrono::steady_clock::now();
            time_part2_ = std::chrono::duration_cast<std::chrono::milliseconds>(time2 - time1).count();

            /// Update obDouble_ using gc_ and gv_ (no phase)
            updateObservation();

            /// AMP goal reward calculation
            // 20210804:1806, Okay now we train AMP
            totalReward_ = exp(-1.0 * 0.25 * pow(target_speed_ - target_direction_.dot(bodyLinearVel_),2));
            assert(totalReward_ > 0);

            /// Reward calculation



            // Third deepmimic reward: end-effector position error by MG
            // head, hands, feet



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

            // 20210804:1806, Okay now we train AMP
            // totalReward_ = angularPosReward_ + angularVelReward_ + endeffPosReward_ + comPosReward_;

            /*
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
            */
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
            Eigen::VectorXd obDouble_amp;
            obDouble_amp.setZero(obDim_amp_);
            anymal_->getState(gc_, gv_);

            int idx_begin = 0;

            raisim::Vec<4> quat;
            raisim::Mat<3, 3> rot{};
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

            ob = obDouble_amp.cast<float>();
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

        void curriculumUpdate() final {
            /*
            if (curriculumFactor_ < 1.0 && curriculumPhase_ == 1)
                curriculumFactor_ += 0.025;
            else
                curriculumFactor_ = 1.0;     */
        }


        /// prototypes of public functions
        // deepmimic functions
        void GetJointInfo();
        void GetScaleFactor(std::string file_path);
        void get_pelvis_velocity(Eigen::VectorXd &pelvis_vel, Eigen::VectorXd &pelvis_vel_ref, Eigen::VectorXd &gv_ref);
        void get_endeffPos_in_two_models(Eigen::MatrixXd &gc_endeff, Eigen::MatrixXd &gc_ref_endeff);
        void get_endeffPos(Eigen::MatrixXd &gc_endeff);
        static raisim::Vec<3> pt_coord_txf(raisim::Mat<3,3> &rot, raisim::Vec<3> &trs, raisim::Vec<3> &pt);
        void set_obDouble_from_gc(Eigen::VectorXd &obDouble, Eigen::VectorXd &gc, int idx_begin);
        int set_gc_from_action(Eigen::VectorXd &targetgc, Eigen::VectorXd &action);
        void set_joint_pos_action_scales(Eigen::VectorXd &actionMean, Eigen::VectorXd &actionStd);
        void set_joint_torque_action_scales(Eigen::VectorXd &actionMean, Eigen::VectorXd &actionStd);
        void set_jointPgain_jointDgain(Eigen::VectorXd &jointPgain, Eigen::VectorXd &jointDgain);

        // muscle functions
        void ReadMuscleModel(const std::string& FileName);
        void ReadMuscleInitialLength(std::string FileName, double *L0);
        void update_seg_txf();

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
        // std::size_t curriculumPhase_ = 0;
        // double curriculumFactor_ = 0.0;

        /// Deepmimic variables
        DeepmimicUtility deepmimic_;
        raisim::ArticulatedSystem *anymal_ref_;
        raisim::ArticulatedSystemVisual *anymal_vis_;
        raisim::World world_ref_;
        std::size_t num_ref_motion_;
        double tau_random_start_;
        double gc_ref_body_y_offset_ = 0.0; // SKOO 20200712 lateral offset of the model
        std::vector<JointInfo> joint_info_;
        std::vector<double> ref_motion_speed_;

        /// Muscle variables
        std::size_t nMuscles_;
        std::vector<MuscleModel> Muscles_;
        std::vector<raisim::Vec<3>> muscle_trs_;
        std::vector<raisim::Mat<3,3>> muscle_rot_;
        std::vector<std::string> scale_link_;
        std::vector<double> scale_value_;

        raisim::PolyLine *Right_contact_;
        raisim::PolyLine *Left_contact_;

        std::size_t info_train_info_on_ = 0;
        std::size_t use_raisim_pdcontrol_on_ = 0;
        std::size_t use_muscle_on_ = 0;
        std::size_t use_direct_torque_on_ = 0;
    };
}

#include "Environment_deepmimic.hpp"
#include "Environment_muscle.hpp"

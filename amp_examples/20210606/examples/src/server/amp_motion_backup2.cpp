// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/RaisimServer.hpp"
#include "raisim/World.hpp"
#include "DeepmimicUtility.hpp"
#include <nlohmann/json.hpp>

void get_endeffPos(Eigen::MatrixXd &gc_endeff);
raisim::Vec<3> pt_coord_txf(raisim::Mat<3,3> &rot, raisim::Vec<3> &trs, raisim::Vec<3> &pt);
void GetJointInfo(std::vector<JointInfo> &joint_info_);
raisim::ArticulatedSystem *anymal_;
nlohmann::json outputs;

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");

  /// create raisim world
  raisim::World world;
  world.setTimeStep(0.001);

  /// create objects
  world.addGround();
  anymal_ = world.addArticulatedSystem(binaryPath.getDirectory() + "\\rsc\\gaitmsk\\urdf\\MSK_GAIT2392_model_joint_template.urdf");

  std::size_t gcDim_ = anymal_->getGeneralizedCoordinateDim();  // 37
  std::size_t gvDim_ = anymal_->getGeneralizedVelocityDim();    // 31
  std::size_t nJoints_ = gvDim_ - 6; // 31-6 = 25

  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  DeepmimicUtility deepmimic_;
  std::vector<JointInfo> joint_info_;

  GetJointInfo(joint_info_);
  deepmimic_.set_joint_info(joint_info_);  // copy the same information to deepmimic operators
  deepmimic_.set_gcdim_gvdim_njoints(gcDim_, gvDim_, nJoints_); // 37, 31, 25

  deepmimic_.add_ref_motion(binaryPath.getDirectory() + "\\rsc\\gaitmsk\\refmotion\\mingi_walk_v2.txt");
  deepmimic_.add_ref_motion(binaryPath.getDirectory() + "\\rsc\\gaitmsk\\refmotion\\mingi_walk_slow_v2.txt");
  deepmimic_.add_ref_motion(binaryPath.getDirectory() + "\\rsc\\gaitmsk\\refmotion\\mingi_walk_fast_v2.txt");
  deepmimic_.add_ref_motion(binaryPath.getDirectory() + "\\rsc\\gaitmsk\\refmotion\\mingi_run_v2.txt");

  int json_count_ = 0;
  for (int irefmotion=0; irefmotion < deepmimic_.get_num_refmotion(); irefmotion++) {
      deepmimic_.set_current_ref_motion_index(irefmotion);

      double refmotion_duration_in_second = deepmimic_.get_duration_ref_motion();
      auto nsample= (std::size_t) (refmotion_duration_in_second / control_dt_);
      for (std::size_t isample=1; isample < nsample; isample++) {
          std::vector<double> state;

          double tau_sample = (isample-1)  * control_dt_;

          Eigen::VectorXd gc0(gcDim_), gc1(gcDim_), gc2(gcDim_), gv0(gvDim_);
          deepmimic_.get_gc_ref_motion(gc0, tau_sample);
          deepmimic_.get_gc_ref_motion(gc1, tau_sample - 1*simulation_dt_);
          deepmimic_.get_gc_ref_motion(gc2, tau_sample + 1*simulation_dt_);
          deepmimic_.get_gv_ref_motion(gv0, gc1, gc2, 2*simulation_dt_);

          anymal_->setState(gc0, gv0);

          std::size_t n_endeff = 5;
          Eigen::MatrixXd gc_endeff(n_endeff, 3);
          get_endeffPos(gc_endeff);

          // /// local root vel calculation with local frame all
          // raisim::Vec<3> local_root_linvel, local_root_angvel;
          // local_root_linvel[0] = gv0[0]; local_root_linvel[1] = gv0[1]; local_root_linvel[2] = gv0[2];
          // local_root_angvel[0] = gv0[3]; local_root_angvel[1] = gv0[4]; local_root_angvel[2] = gv0[5];
          //
          // raisim::Mat<3, 3> pelvis_R{};
          // anymal_->getBaseOrientation(pelvis_R);
          // raisim::Mat<3, 3> pelvis_R_inverse = pelvis_R.transpose();
          //
          // local_root_linvel = pelvis_R_inverse * local_root_linvel;
          // local_root_angvel = pelvis_R_inverse * local_root_angvel;

          /// local root vel calculation with observe method
          raisim::Vec<4> quat;
          raisim::Mat<3, 3> rot;
          quat[0] = gc0[3];
          quat[1] = gc0[4];
          quat[2] = gc0[5];
          quat[3] = gc0[6];
          raisim::quatToRotMat(quat, rot);

          Eigen::Vector3d bodyLinearVel, bodyAngularVel;
          bodyLinearVel = rot.e().transpose() * gv0.segment(0, 3);
          bodyAngularVel = rot.e().transpose() * gv0.segment(3, 3);


          for(int i=0; i < bodyLinearVel.size(); i++){
            state.push_back(bodyLinearVel[i]);
          }
          for(int i=0; i < bodyAngularVel.size(); i++){
            state.push_back(bodyAngularVel[i]);
          }
          for(int i=7; i < gc0.size(); i++){
            state.push_back(gc0[i]);
          }
          for(int i=6; i < gv0.size(); i++){
            state.push_back(gv0[i]);
          }
          for(int i=0; i < gc_endeff.size(); i++){
            state.push_back(gc_endeff(i));
          }

          // std::cout<<prev_state.size()<<std::endl;

          // outputs["expert"][json_count_] = prev_state;

          tau_sample = isample * control_dt_;

          // Eigen::VectorXd gc0(gcDim_), gc1(gcDim_), gc2(gcDim_), gv0(gvDim_);
          deepmimic_.get_gc_ref_motion(gc0, tau_sample);
          deepmimic_.get_gc_ref_motion(gc1, tau_sample - 1*simulation_dt_);
          deepmimic_.get_gc_ref_motion(gc2, tau_sample + 1*simulation_dt_);
          deepmimic_.get_gv_ref_motion(gv0, gc1, gc2, 2*simulation_dt_);

          anymal_->setState(gc0, gv0);

          // std::size_t n_endeff = 5;
          // Eigen::MatrixXd gc_endeff(n_endeff, 3);
          get_endeffPos(gc_endeff);

          // /// local root vel calculation with local frame all
          // raisim::Vec<3> local_root_linvel, local_root_angvel;
          // local_root_linvel[0] = gv0[0]; local_root_linvel[1] = gv0[1]; local_root_linvel[2] = gv0[2];
          // local_root_angvel[0] = gv0[3]; local_root_angvel[1] = gv0[4]; local_root_angvel[2] = gv0[5];
          //
          // raisim::Mat<3, 3> pelvis_R{};
          // anymal_->getBaseOrientation(pelvis_R);
          // raisim::Mat<3, 3> pelvis_R_inverse = pelvis_R.transpose();
          //
          // local_root_linvel = pelvis_R_inverse * local_root_linvel;
          // local_root_angvel = pelvis_R_inverse * local_root_angvel;

          /// local root vel calculation with observe method
          // raisim::Vec<4> quat;
          // raisim::Mat<3, 3> rot;
          quat[0] = gc0[3];
          quat[1] = gc0[4];
          quat[2] = gc0[5];
          quat[3] = gc0[6];
          raisim::quatToRotMat(quat, rot);

          // Eigen::Vector3d bodyLinearVel, bodyAngularVel;
          bodyLinearVel = rot.e().transpose() * gv0.segment(0, 3);
          bodyAngularVel = rot.e().transpose() * gv0.segment(3, 3);


          for(int i=0; i < bodyLinearVel.size(); i++){
            state.push_back(bodyLinearVel[i]);
          }
          for(int i=0; i < bodyAngularVel.size(); i++){
            state.push_back(bodyAngularVel[i]);
          }
          for(int i=7; i < gc0.size(); i++){
            state.push_back(gc0[i]);
          }
          for(int i=6; i < gv0.size(); i++){
            state.push_back(gv0[i]);
          }
          for(int i=0; i < gc_endeff.size(); i++){
            state.push_back(gc_endeff(i));
          }

          std::cout<<state.size()<<std::endl;

          outputs["expert"][json_count_] = state;

          json_count_ += 1;



          // if(irefmotion == 3 && isample == 62){
          //   // std::cout<<gv0<<std::endl;
          //   // std::cout<<gc_endeff.row(0)<<std::endl;
          //   // std::cout<<local_root_linvel<<std::endl;
          //   // std::cout<<local_root_angvel<<std::endl;
          //
          //   // std::cout<<bodyLinearVel<<std::endl;
          //   // std::cout<<bodyAngularVel<<std::endl;
          //
          //   std::cout<<gc0.size()<<std::endl;
          //   std::cout<<gv0.size()<<std::endl;
          //
          //   std::cout<<bodyLinearVel.size()<<std::endl;
          //   std::cout<<bodyAngularVel.size()<<std::endl;
          //
          //   std::cout<<gc_endeff.size()<<std::endl;
          // }

          // std::cout<<"irefmotion: "<<irefmotion<<", isample: "<<isample<<std::endl;
      }
  }
  std::ofstream o("example.json");
  o << std::setw(4) << outputs << std::endl;

  /// launch raisim servear
  raisim::RaisimServer server(&world);
  server.launchServer();
  server.focusOn(anymal_);

  for (int i=0; i<200000000; i++) {
    std::this_thread::sleep_for(std::chrono::microseconds(1000));
    // server.integrateWorldThreadSafe();
  }

  server.killServer();
}

void get_endeffPos(Eigen::MatrixXd &gc_endeff) {
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

raisim::Vec<3> pt_coord_txf(raisim::Mat<3,3> &rot, raisim::Vec<3> &trs, raisim::Vec<3> &pt) {
    raisim::Vec<3> pt_txfed = {
            rot[0] * pt[0] + rot[3] * pt[1] + rot[6] * pt[2] + trs[0],
            rot[1] * pt[0] + rot[4] * pt[1] + rot[7] * pt[2] + trs[1],
            rot[2] * pt[0] + rot[5] * pt[1] + rot[8] * pt[2] + trs[2]
    };

    return pt_txfed;
}

void GetJointInfo(std::vector<JointInfo> &joint_info_) {
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

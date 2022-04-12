namespace raisim {
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


    void ENVIRONMENT::get_pelvis_velocity(Eigen::VectorXd &pelvis_vel, Eigen::VectorXd &pelvis_vel_ref, Eigen::VectorXd &gv_ref) {
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


    void ENVIRONMENT::get_endeffPos_in_two_models(Eigen::MatrixXd &gc_endeff, Eigen::MatrixXd &gc_ref_endeff) {
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


    raisim::Vec<3> ENVIRONMENT::pt_coord_txf(raisim::Mat<3,3> &rot, raisim::Vec<3> &trs, raisim::Vec<3> &pt) {
        raisim::Vec<3> pt_txfed = {
                rot[0] * pt[0] + rot[3] * pt[1] + rot[6] * pt[2] + trs[0],
                rot[1] * pt[0] + rot[4] * pt[1] + rot[7] * pt[2] + trs[1],
                rot[2] * pt[0] + rot[5] * pt[1] + rot[8] * pt[2] + trs[2]
        };

        return pt_txfed;
    }


    int ENVIRONMENT::set_gc_from_action(Eigen::VectorXd &targetgc, Eigen::VectorXd &action) {
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


    void ENVIRONMENT::set_obDouble_from_gc(Eigen::VectorXd &obDouble, Eigen::VectorXd &gc, int idx_begin) {
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


    void ENVIRONMENT::set_joint_pos_action_scales(Eigen::VectorXd &actionMean, Eigen::VectorXd &actionStd) {
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


    void ENVIRONMENT::set_joint_torque_action_scales(Eigen::VectorXd &actionMean, Eigen::VectorXd &actionStd) {
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


    void ENVIRONMENT::set_jointPgain_jointDgain(Eigen::VectorXd &jointPgain, Eigen::VectorXd &jointDgain) {
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
}
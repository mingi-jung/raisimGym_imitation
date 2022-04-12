//
// Created by skoo on 20. 7. 2..
//

#ifndef _RAISIM_GYM_DEEPMIMICUTILITY_H
#define _RAISIM_GYM_DEEPMIMICUTILITY_H

#include <cmath>
#include <random>
#include <nlohmann/json.hpp>
#include "quaternion_mskbiodyn.hpp"


class JointInfo {
public:
    JointInfo() {}
    ~JointInfo() {}
    void set_name_type_gc_gv(std::string name, std::string type, int ngc, int ngv) {m_name=name; m_type=type; m_ngc=ngc; m_ngv=ngv;}
    void set_gcidx(int gc_index_0, int gc_index_1, int gc_index_2, int gc_index_3) {m_gc_index[0]=gc_index_0; m_gc_index[1]=gc_index_1; m_gc_index[2]=gc_index_2; m_gc_index[3]=gc_index_3;}
    void set_gvidx(int gv_index_0, int gv_index_1, int gv_index_2) {m_gv_index[0]=gv_index_0; m_gv_index[1]=gv_index_1; m_gv_index[2]=gv_index_2;}
    void set_pos_actionmean(double action_mean_0, double action_mean_1, double action_mean_2) {m_pos_action_mean[0]=action_mean_0; m_pos_action_mean[1]=action_mean_1; m_pos_action_mean[2]=action_mean_2;}
    void set_pos_actionstd(double action_std_0, double action_std_1, double action_std_2) {m_pos_action_std[0]=action_std_0; m_pos_action_std[1]=action_std_1; m_pos_action_std[2]=action_std_2;}
    void set_vel_actionmean(double action_mean_0, double action_mean_1, double action_mean_2) {m_vel_action_mean[0]=action_mean_0; m_vel_action_mean[1]=action_mean_1; m_vel_action_mean[2]=action_mean_2;}
    void set_vel_actionstd(double action_std_0, double action_std_1, double action_std_2) {m_vel_action_std[0]=action_std_0; m_vel_action_std[1]=action_std_1; m_vel_action_std[2]=action_std_2;}
    void set_torque_actionmean(double action_mean_0, double action_mean_1, double action_mean_2) {m_torque_action_mean[0]=action_mean_0; m_torque_action_mean[1]=action_mean_1; m_torque_action_mean[2]=action_mean_2;}
    void set_torque_actionstd(double action_std_0, double action_std_1, double action_std_2) {m_torque_action_std[0]=action_std_0; m_torque_action_std[1]=action_std_1; m_torque_action_std[2]=action_std_2;}
    void set_pgain(double pgain_0, double pgain_1, double pgain_2) {m_pgain[0]=pgain_0; m_pgain[1]=pgain_1; m_pgain[2]=pgain_2;}
    void set_dgain(double dgain_0, double dgain_1, double dgain_2) {m_dgain[0]=dgain_0; m_dgain[1]=dgain_1; m_dgain[2]=dgain_2;}
    void set_ref_coord_index(int ref_index_0, int ref_index_1, int ref_index_2, int ref_index_3) {m_ref_coord_index[0]=ref_index_0; m_ref_coord_index[1]=ref_index_1; m_ref_coord_index[2]=ref_index_2; m_ref_coord_index[3]=ref_index_3;}

    std::string get_name() {return m_name;}
    std::string get_joint_type() {return m_type;}
    int get_ngc() const {return m_ngc;}
    int get_ngv() const {return m_ngv;}
    void get_gc_index(int *gc_index) {gc_index[0]=m_gc_index[0]; gc_index[1]=m_gc_index[1]; gc_index[2]=m_gc_index[2]; gc_index[3]=m_gc_index[3];}
    void get_gv_index(int *gv_index) {gv_index[0]=m_gv_index[0]; gv_index[1]=m_gv_index[1]; gv_index[2]=m_gv_index[2];}
    void get_pos_action_mean(double *action_mean) {action_mean[0]=m_pos_action_mean[0]; action_mean[1]=m_pos_action_mean[1]; action_mean[2]=m_pos_action_mean[2];}
    void get_pos_action_std(double *action_std) {action_std[0]=m_pos_action_std[0]; action_std[1]=m_pos_action_std[1]; action_std[2]=m_pos_action_std[2];}
    void get_vel_action_mean(double *action_mean) {action_mean[0]=m_vel_action_mean[0]; action_mean[1]=m_vel_action_mean[1]; action_mean[2]=m_vel_action_mean[2];}
    void get_vel_action_std(double *action_std) {action_std[0]=m_vel_action_std[0]; action_std[1]=m_vel_action_std[1]; action_std[2]=m_vel_action_std[2];}
    void get_torque_action_mean(double *action_mean) {action_mean[0]=m_torque_action_mean[0]; action_mean[1]=m_torque_action_mean[1]; action_mean[2]=m_torque_action_mean[2];}
    void get_torque_action_std(double *action_std) {action_std[0]=m_torque_action_std[0]; action_std[1]=m_torque_action_std[1]; action_std[2]=m_torque_action_std[2];}
    void get_pgain(double *pgain) {pgain[0]=m_pgain[0]; pgain[1]=m_pgain[1]; pgain[2]=m_pgain[2];}
    void get_dgain(double *dgain) {dgain[0]=m_dgain[0]; dgain[1]=m_dgain[1]; dgain[2]=m_dgain[2];}
    void get_ref_coord_index(int *ref_index) {ref_index[0]=m_ref_coord_index[0]; ref_index[1]=m_ref_coord_index[1]; ref_index[2]=m_ref_coord_index[2]; ref_index[3]=m_ref_coord_index[3];}

private:
    std::string m_name;
    std::string m_type;
    int m_gc_index[4];
    int m_gv_index[3];
    int m_ngc;
    int m_ngv;
    int m_ref_coord_index[4];
    double m_pos_action_mean[3];
    double m_pos_action_std[3];
    double m_vel_action_mean[3];
    double m_vel_action_std[3];
    double m_torque_action_mean[3];
    double m_torque_action_std[3];
    double m_pgain[3];
    double m_dgain[3];
};


class RefMotion {
public:
    RefMotion() {}
    ~RefMotion() {}
    void read_ref_motion_from_file(std::string ref_motion_filename, int gcdim, std::vector<JointInfo> &joint_info);
    void get_cycle_tau(double tau_at, int &ncycle, double &tau_in_cycle) const;
    void get_gc_ref_motion(Eigen::VectorXd &ref_motion_one, double tau_at);
    void get_gv_ref_motion(Eigen::VectorXd &gv, Eigen::VectorXd &gc1, Eigen::VectorXd &gc2, double dt);
    double get_phase(double tau_at) const;
    double get_duration_ref_motion() const {return m_totalframetime;}
    double get_ref_body_y_offset() const {return m_ref_body_y_offset;}
    void set_ref_motion_lateral_offset(double body_y_offset);
    bool iscyclic() const {return m_bcyclic;}
    std::size_t get_num_frame() {return m_nframe;}

private:
    void calc_slerp_joint(Eigen::VectorXd &, const int *, int, int, double);
    void calc_interp_joint(Eigen::VectorXd &, int , int, int, double);
    static void calc_rotvel_joint(Eigen::VectorXd &gv, const int *idx_gv, Eigen::VectorXd &gc1, Eigen::VectorXd &gc2, const int *idx_gc, double dt);
    static void calc_linvel_joint(Eigen::VectorXd &gv, int idx_gv, Eigen::VectorXd &gc1, Eigen::VectorXd &gc2, int idx_gc, double dt);

private:
    std::string m_ref_motion_filename;
    std::vector<Eigen::VectorXd> m_ref_motion;
    std::vector<double> m_cumulframetime;

    int m_gcdim;
    bool m_bcyclic = false;
    int m_nframe = -1;
    double m_totalframetime = -1;
    double m_ref_body_y_offset = 0;
    std::vector<JointInfo> m_joint_info;
};


void RefMotion::read_ref_motion_from_file(std::string ref_motion_filename, int gcdim, std::vector<JointInfo> &joint_info)
{
    /// read motion data from a reference file
    m_ref_motion_filename = std::move(ref_motion_filename);
    m_gcdim = gcdim;
    m_joint_info = joint_info;
    std::ifstream infile1(m_ref_motion_filename);      // std::cout << m_ref_motion_file << std::endl;
    nlohmann::json jsondata;
    infile1 >> jsondata;

    /// check if the reference motion is cyclic (wrap) or not
    std::string strloop = jsondata["Loop"];
    std::transform(strloop.begin(), strloop.end(), strloop.begin(), ::tolower);
    if (strloop == "wrap")
        m_bcyclic = true;

    /// joint index conversion between raisim model and motion data
    Eigen::VectorXd jointIdxConvTable(m_gcdim); // 37
    int ngc, idx_begin = 0;
    int ref_coord_index[4];

    for (int ijoint=0; ijoint<m_joint_info.size(); ijoint++) {
        ngc = m_joint_info[ijoint].get_ngc();
        m_joint_info[ijoint].get_ref_coord_index(ref_coord_index);
        for (int igc=0; igc<ngc; igc++) {
            jointIdxConvTable(idx_begin+igc) = ref_coord_index[igc];
        }
        idx_begin += ngc;
    }

    int nframe = jsondata["Frames"].size();  // m_ref_motion.clear(); // m_cumulframetime.clear();
    m_ref_motion.resize(nframe);
    m_cumulframetime.resize(nframe);

    Eigen::VectorXd jointNominalConfig(m_gcdim);  // variable to obtain gc of a frame
    double totaltime = 0;                         // variable to accumulate time of frames

    for (int iframe = 0; iframe < nframe; iframe++) {
        jointNominalConfig.setZero();
        for (int igcjoint = 0; igcjoint < m_gcdim; igcjoint++)
            jointNominalConfig[igcjoint] = jsondata["Frames"][iframe][jointIdxConvTable[igcjoint]];

        // rotate the root body by 90 degree along the x-axis
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
        // END of root body 90 degree rotation

        m_ref_motion[iframe] = jointNominalConfig;

        // SKOO 20200706 Force that all frames have duration
        totaltime += (double) jsondata["Frames"][iframe][0];
        m_cumulframetime[iframe] = totaltime;
    }

    m_nframe = nframe;
    m_totalframetime = totaltime;
}


void RefMotion::set_ref_motion_lateral_offset(double body_y_offset)
{
    for (int iframe = 0; iframe < m_nframe; iframe++) {
        m_ref_motion[iframe](1) -= body_y_offset;
    }
    m_ref_body_y_offset = body_y_offset;
}


void RefMotion::get_gv_ref_motion(Eigen::VectorXd &gv, Eigen::VectorXd &gc1, Eigen::VectorXd &gc2, double dt)
{
    std::string joint_type;
    int gc_index[4], gv_index[3];
    int ngv, idx_begin = 0;

    for (int ijoint=0; ijoint<m_joint_info.size(); ijoint++) {
        m_joint_info[ijoint].get_gc_index(gc_index);
        m_joint_info[ijoint].get_gv_index(gv_index);
        joint_type = m_joint_info[ijoint].get_joint_type();
        ngv = m_joint_info[ijoint].get_ngv();

        if (joint_type == "rot3") {
            calc_rotvel_joint(gv, gv_index, gc1, gc2, gc_index, dt);
        }
        if (joint_type == "rot1") {
            calc_linvel_joint(gv, gv_index[0], gc1, gc2, gc_index[0], dt);
        }
        if (joint_type == "trs3") {
            calc_linvel_joint(gv, gv_index[0], gc1, gc2, gc_index[0], dt);
            calc_linvel_joint(gv, gv_index[1], gc1, gc2, gc_index[1], dt);
            calc_linvel_joint(gv, gv_index[2], gc1, gc2, gc_index[2], dt);
        }
        idx_begin += ngv;
    }
}


void RefMotion::get_gc_ref_motion(Eigen::VectorXd &ref_motion_one, double tau_at)
{
    int ncycle;
    int idx_frame_prev = -1;
    int idx_frame_next = -1;

    double tau;
    double t_offset_base;
    double t_interval, t_interp;

    get_cycle_tau(tau_at, ncycle, tau);

    for (int iframe = 0; iframe < m_nframe; iframe++) {
        if (tau < m_cumulframetime[iframe]) { // SKOO 20200706 equal is checked
            idx_frame_prev = iframe;
            idx_frame_next = iframe + 1;
            break;
        }
    }
    if (idx_frame_next >= m_nframe) {
        idx_frame_next = 0;
    }

    // std::cout << m_nframe << " : " << idx_frame_prev << " : " << idx_frame_next << std::endl;
    // assert(idx_frame_prev >= 0); // SKOO 20200707 assert does not seem to work here
    // assert(idx_frame_prev < m_nframe); // SKOO 20200707 assert does not seem to work here

    if (idx_frame_prev == 0)
        t_offset_base = 0;
    else
        t_offset_base = m_cumulframetime[idx_frame_prev-1];

    t_interval = m_cumulframetime[idx_frame_prev] - t_offset_base;
    t_interp = (tau - t_offset_base) / t_interval;

    std::string joint_type;
    int ngc, gc_index[4];
    int idx_begin = 0;

    for (int ijoint=0; ijoint<m_joint_info.size(); ijoint++) {
        m_joint_info[ijoint].get_gc_index(gc_index);
        joint_type = m_joint_info[ijoint].get_joint_type();
        ngc = m_joint_info[ijoint].get_ngc();

        if (joint_type == "rot3") {
            calc_slerp_joint(ref_motion_one, gc_index, idx_frame_prev, idx_frame_next, t_interp);
        }
        if (joint_type == "rot1") {
            calc_interp_joint(ref_motion_one, gc_index[0], idx_frame_prev, idx_frame_next, t_interp);
        }
        if (joint_type == "trs3") {
            calc_interp_joint(ref_motion_one, gc_index[0], idx_frame_prev, idx_frame_next, t_interp);
            calc_interp_joint(ref_motion_one, gc_index[1], idx_frame_prev, idx_frame_next, t_interp);
            calc_interp_joint(ref_motion_one, gc_index[2], idx_frame_prev, idx_frame_next, t_interp);
        }
        idx_begin += ngc;
    }

    if (m_bcyclic) {
        double xgap   = (m_ref_motion[m_nframe - 1][0] - m_ref_motion[0][0]) / (m_nframe - 1);
        double x_displacement_per_cycle_including_gap = (m_ref_motion[m_nframe - 1][0] - m_ref_motion[0][0]) + xgap;
        ref_motion_one[0] += x_displacement_per_cycle_including_gap * ncycle; // 20200706 SKOO check it again?
    }
}


void RefMotion::get_cycle_tau(double tau_at, int &ncycle, double &tau_in_cycle) const
{
    ncycle = 0;
    tau_in_cycle = tau_at;

    // In case that the reference motion is cyclic
    if (m_bcyclic) {
        if (tau_at > m_totalframetime) {  // SKOO 20200706 check when equal?
            ncycle = (int) (tau_at / m_totalframetime);
            tau_in_cycle = tau_at - m_totalframetime * ncycle;
        }
    } else {
        // assert(tau < m_totalframetime); // SKOO 20200706 check when equal?
        if (tau_in_cycle > m_totalframetime)
            std::cout << "Error : ref motion time over (" << tau_in_cycle << ", " << m_totalframetime << ")" << std::endl;
    }
    // std::cout << m_totalframetime << " : " << tau_in << " : " << ncycle << " : " << tau << std::endl;
}


void RefMotion::calc_slerp_joint(
        Eigen::VectorXd &ref_motion_one,          // output
        const int *idxlist,                       // indices to be updated
        int idxprev,                              // input
        int idxnext,                              // input
        double t_interp) {                        // input

    int idxW = idxlist[0];
    int idxX = idxlist[1];
    int idxY = idxlist[2];
    int idxZ = idxlist[3];

    Quaternion qprev(m_ref_motion[idxprev][idxW], m_ref_motion[idxprev][idxX],
                     m_ref_motion[idxprev][idxY], m_ref_motion[idxprev][idxZ]);
    Quaternion qnext(m_ref_motion[idxnext][idxW], m_ref_motion[idxnext][idxX],
                     m_ref_motion[idxnext][idxY], m_ref_motion[idxnext][idxZ]);
    Quaternion qout;

    Quaternion::slerp(qout, qprev, qnext, t_interp);
    ref_motion_one[idxW] = qout.getW();
    ref_motion_one[idxX] = qout.getX();
    ref_motion_one[idxY] = qout.getY();
    ref_motion_one[idxZ] = qout.getZ();
}


void RefMotion::calc_interp_joint(
        Eigen::VectorXd &ref_motion_one,   // output
        int idxjoint,                      // index of a scalar joint to be updated
        int idxprev,                       // input
        int idxnext,                       // input
        double t_interp) {                 // input

    double a = m_ref_motion[idxprev][idxjoint];
    double b = m_ref_motion[idxnext][idxjoint];
    double xgap;

    // SKOO 20200707 interpolation of body xpos at the end of frame
    if (idxnext == 0 && idxjoint == 0) {
        xgap = (m_ref_motion[m_nframe - 1][0] - m_ref_motion[0][0]) / (m_nframe - 1);
        b = a + xgap;
    }

    ref_motion_one[idxjoint] = a + (b - a) * t_interp;
}


void RefMotion::calc_linvel_joint(
        Eigen::VectorXd &gv,    // output angular velocity
        const int idx_gv,
        Eigen::VectorXd &gc1,
        Eigen::VectorXd &gc2,
        const int idx_gc,
        double dt)
{
    gv(idx_gv) = (gc2(idx_gc) - gc1(idx_gc)) / dt;
}


void RefMotion::calc_rotvel_joint(
        Eigen::VectorXd &gv,    // output angular velocity
        const int *idx_gv,
        Eigen::VectorXd &gc1,
        Eigen::VectorXd &gc2,
        const int *idx_gc,
        double dt)
{
    // [average angular velocity] = 2*log(q2*inv(q1))/dt
    Quaternion q1(gc1(idx_gc[0]), gc1(idx_gc[1]), gc1(idx_gc[2]), gc1(idx_gc[3]));
    Quaternion q2(gc2(idx_gc[0]), gc2(idx_gc[1]), gc2(idx_gc[2]), gc2(idx_gc[3]));

    Quaternion q3 = q2*q1.inv();
    Quaternion q4 = q3.log()*2/dt;

    gv(idx_gv[0]) = q4.getX();
    gv(idx_gv[1]) = q4.getY();
    gv(idx_gv[2]) = q4.getZ();
}


double RefMotion::get_phase(double tau_at) const
{
    int ncycle;
    double tau;

    get_cycle_tau(tau_at, ncycle, tau);
    double phase = tau/m_totalframetime; // [0, 1]

    return phase;
}


class DeepmimicUtility {
public:
    DeepmimicUtility() {}
    ~DeepmimicUtility() {}

    void add_ref_motion(std::string ref_motion_filename);
    void set_joint_info(std::vector<JointInfo> &joint_info) {m_joint_info = joint_info;}
    void set_ref_motion_lateral_offset(double body_y_offset) {m_refmotions[m_current_ref_motion_index].set_ref_motion_lateral_offset(body_y_offset);}
    void set_current_ref_motion_index(int ref_motion_index);
    void set_current_ref_motion_random();
    void set_gcdim_gvdim_njoints(int gcDim, int gvDim, int nJoints) {m_gcdim = gcDim; /*m_gvdim = gvDim; m_njoints = nJoints;*/}

    void get_gc_ref_motion(Eigen::VectorXd &ref_motion_one, double tau_at);
    void get_gv_ref_motion(Eigen::VectorXd &gv, Eigen::VectorXd &gc1, Eigen::VectorXd &gc2, double dt);
    double get_phase(double tau_at) {return m_refmotions[m_current_ref_motion_index].get_phase(tau_at);};
    double get_tau_random_start();
    double get_duration_ref_motion() {return m_refmotions[m_current_ref_motion_index].get_duration_ref_motion();}
    bool iscyclic() {return m_refmotions[m_current_ref_motion_index].iscyclic();}
    std::size_t get_num_refmotion() {return m_refmotions.size();}
    std::size_t get_num_frame() {return m_refmotions[m_current_ref_motion_index].get_num_frame();}

    double get_angularPosReward(Eigen::VectorXd &gc, Eigen::VectorXd &gc_ref);
    double get_angularVelReward(Eigen::VectorXd &gv, Eigen::VectorXd &gv_ref);
    double get_endeffPosReward(int n_endeff, Eigen::MatrixXd &gc_endeff, Eigen::MatrixXd &gc_ref_endeff);
    double get_comPosReward(Eigen::VectorXd &vcom1, Eigen::VectorXd &vcom2);
    // double get_rootReward();

private:
    std::vector<JointInfo> m_joint_info;
    std::vector<RefMotion> m_refmotions;

    double m_root_pos_err, m_root_rot_err, m_root_vel_err, m_root_ang_vel_err;
    int m_gcdim;
    int m_current_ref_motion_index = 0;
};


void DeepmimicUtility::add_ref_motion(std::string ref_motion_filename) {
    RefMotion refmotion;
    refmotion.read_ref_motion_from_file(std::move(ref_motion_filename), m_gcdim, m_joint_info);
    this->m_refmotions.push_back(refmotion);
}


void DeepmimicUtility::set_current_ref_motion_index(int ref_motion_index)
{
    m_current_ref_motion_index = ref_motion_index;
};


void DeepmimicUtility::set_current_ref_motion_random()
{
    std::random_device rd;  // random device
    std::mt19937 mersenne(rd()); // random generator, a mersenne twister
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    double random_number = distribution(mersenne);

    m_current_ref_motion_index = (int)((this->m_refmotions.size()-0.000001) * random_number);
}


void DeepmimicUtility::get_gv_ref_motion(Eigen::VectorXd &gv, Eigen::VectorXd &gc1, Eigen::VectorXd &gc2, double dt)
{
    m_refmotions[m_current_ref_motion_index].get_gv_ref_motion(gv, gc1, gc2, dt);
}


void DeepmimicUtility::get_gc_ref_motion(Eigen::VectorXd &ref_motion_one, double tau_at)
{
    m_refmotions[m_current_ref_motion_index].get_gc_ref_motion(ref_motion_one, tau_at);
}


double DeepmimicUtility::get_tau_random_start()
{
    std::random_device rd;  // random device
    std::mt19937 mersenne(rd()); // random generator, a mersenne twister
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    double random_number = distribution(mersenne);
    double tau_random_start = 0;

    if (m_refmotions[m_current_ref_motion_index].iscyclic())
        tau_random_start = m_refmotions[m_current_ref_motion_index].get_duration_ref_motion() * 0.9 * random_number;
    else
        tau_random_start = m_refmotions[m_current_ref_motion_index].get_duration_ref_motion() * 0.5 * random_number;

    return tau_random_start;
}


double DeepmimicUtility::get_angularPosReward(Eigen::VectorXd &gc, Eigen::VectorXd &gc_ref)
{
    Quaternion q1, q2;
    std::vector<double> vecdiffpos;
    vecdiffpos.clear();

    /*
    bool is_nan = false;
    for (int i=0; i<43; i++) {
        if (isnan(gc_ref(i))) {
            is_nan = true;
            break;
        }
    }
    if (is_nan) {
        for (int i = 0; i < 43; ++i) std::cout << gc_ref(i) << ' '; std::cout << std::endl;
        std::cout << "nan in gc_ref_ in get_angularPosReward; escaping nan" << std::endl;
        return NAN;
    }
    */

    /// root body pose error calculation
    int idx_root[3]       = {0, 1, 2};
    int idx_qroot[4]      = {3, 4, 5, 6};

    double ref_body_y_offset = m_refmotions[m_current_ref_motion_index].get_ref_body_y_offset();

    q1.set(gc(idx_qroot[0]), gc(idx_qroot[1]), gc(idx_qroot[2]), gc(idx_qroot[3]));
    q2.set(gc_ref(idx_qroot[0]), gc_ref(idx_qroot[1]), gc_ref(idx_qroot[2]), gc_ref(idx_qroot[3]));
    m_root_rot_err = Quaternion::get_angle_two_quaternions(q1, q2);

    m_root_pos_err = sqrt(
            (gc(idx_root[0])  - gc_ref(idx_root[0])) * (gc(idx_root[0])  - gc_ref(idx_root[0]))
            + (gc(idx_root[1])  - (gc_ref(idx_root[1])+ref_body_y_offset)) * (gc(idx_root[1])  - (gc_ref(idx_root[1])+ref_body_y_offset))
            + (gc(idx_root[2])  - gc_ref(idx_root[2])) * (gc(idx_root[2])  - gc_ref(idx_root[2])));

    /// joint position error calculation
    std::string joint_type;
    int ngc, gc_index[4];
    int idx_begin = 0;

    // SKOO 20200708 should we include the base orientation and position? Not sure...
    for (int ijoint=2; ijoint<m_joint_info.size(); ijoint++) {  // excluding root trs and rot
        m_joint_info[ijoint].get_gc_index(gc_index);
        joint_type = m_joint_info[ijoint].get_joint_type();
        ngc = m_joint_info[ijoint].get_ngc();

        if (joint_type == "rot3") {
            q1.set(gc(gc_index[0]), gc(gc_index[1]), gc(gc_index[2]), gc(gc_index[3]));
            q2.set(gc_ref(gc_index[0]), gc_ref(gc_index[1]), gc_ref(gc_index[2]), gc_ref(gc_index[3]));
            vecdiffpos.push_back(Quaternion::get_angle_two_quaternions(q1, q2));
        }
        if (joint_type == "rot1") {
            vecdiffpos.push_back(abs(gc(gc_index[0]) - gc_ref(gc_index[0])));
        }
        if (joint_type == "trs3") {
            vecdiffpos.push_back(abs(gc(gc_index[0]) - gc_ref(gc_index[0])));
            vecdiffpos.push_back(abs(gc(gc_index[1]) - gc_ref(gc_index[1])));
            vecdiffpos.push_back(abs(gc(gc_index[2]) - gc_ref(gc_index[2])));
        }
        idx_begin += ngc;
    }

    /// calculate the metric for the joint position error
    double sumsquare_diffpos = 0;
    for (int i=0; i< vecdiffpos.size(); ++i) {
        sumsquare_diffpos += vecdiffpos[i] * vecdiffpos[i];
        // if (std::isnan(vecdiffpos[i]))
        //     std::cout << "vecdiffpos[" << i << "] :" << vecdiffpos[i] << ' ' << std::endl;
    }
    // std::cout << std::endl;

    /* In original deepmimic
	const double pose_scale = 2.0 / 15 * num_joints;
	const double err_scale = 1;
    double pose_reward = exp(-err_scale * pose_scale * pose_err);
    */
    // int num_joints = 10 + 6; // revolute (10), spherical (6)
    // const double pose_scale = 2.0 / 15 * num_joints;
    // const double err_scale = 1;
    // double pose_reward = exp(-err_scale * pose_scale * sumdiffpos);

    /*
    // SKOO 20200709 Debugging nan
    if (std::isnan(pose_reward)) {
        std::cout << "pose_reward is nan!" << std::endl;

        for (int i=0; i< 43; ++i)
            std::cout << gc(i) << ' ';
        std::cout << std::endl;
        for (int i=0; i< 43; ++i)
            std::cout << gc_ref(i) << ' ';

        // assert(!isnan(pose_reward));
    }
    */

    // return pose_reward;
    return sumsquare_diffpos;
}


double DeepmimicUtility::get_angularVelReward(Eigen::VectorXd &gv, Eigen::VectorXd &gv_ref)
{
    std::vector<double> vecdiffvel;
    vecdiffvel.clear();

    /// root body velocity error calculation
    int idx_gv_root[3]       = {0, 1, 2};
    int idx_gv_qroot[3]      = {3, 4, 5};

    m_root_vel_err = sqrt(
            (gv(idx_gv_root[0]) - gv_ref(idx_gv_root[0])) * (gv(idx_gv_root[0]) - gv_ref(idx_gv_root[0]))
            + (gv(idx_gv_root[1]) - gv_ref(idx_gv_root[1])) * (gv(idx_gv_root[1]) - gv_ref(idx_gv_root[1]))
            + (gv(idx_gv_root[2]) - gv_ref(idx_gv_root[2])) * (gv(idx_gv_root[2]) - gv_ref(idx_gv_root[2])));

    m_root_ang_vel_err = sqrt(
            (gv(idx_gv_qroot[0]) - gv_ref(idx_gv_qroot[0])) * (gv(idx_gv_qroot[0]) - gv_ref(idx_gv_qroot[0]))
            + (gv(idx_gv_qroot[1]) - gv_ref(idx_gv_qroot[1])) * (gv(idx_gv_qroot[1]) - gv_ref(idx_gv_qroot[1]))
            + (gv(idx_gv_qroot[2]) - gv_ref(idx_gv_qroot[2])) * (gv(idx_gv_qroot[2]) - gv_ref(idx_gv_qroot[2])));

    // for (int i=0; i< 34; ++i) std::cout << gv(i) << ' '; std::cout << std::endl;
    // for (int i=0; i< 34; ++i) std::cout << gv_ref(i) << ' '; std::cout << std::endl;

    // for (int idx=3; idx<31; idx++) {  // 31???
    //     vecdiffvel.push_back((gv(idx) - gv_ref(idx)));
    // }

    /// joint velocity error calculation
    std::string joint_type;
    int ngv, gv_index[3];
    int idx_begin = 0;

    for (int ijoint=2; ijoint<m_joint_info.size(); ijoint++) {  // excluding root trs and rot
        m_joint_info[ijoint].get_gv_index(gv_index);
        joint_type = m_joint_info[ijoint].get_joint_type();
        ngv = m_joint_info[ijoint].get_ngv();

        if (joint_type == "rot3") {
            vecdiffvel.push_back(gv(gv_index[0]) - gv_ref(gv_index[0]));
            vecdiffvel.push_back(gv(gv_index[1]) - gv_ref(gv_index[1]));
            vecdiffvel.push_back(gv(gv_index[2]) - gv_ref(gv_index[2]));
        }
        if (joint_type == "rot1") {
            vecdiffvel.push_back(gv(gv_index[0]) - gv_ref(gv_index[0]));
        }
        if (joint_type == "trs3") {
            vecdiffvel.push_back(gv(gv_index[0]) - gv_ref(gv_index[0]));
            vecdiffvel.push_back(gv(gv_index[1]) - gv_ref(gv_index[1]));
            vecdiffvel.push_back(gv(gv_index[2]) - gv_ref(gv_index[2]));
        }
        idx_begin += ngv;
    }

    /// calculate the metric for the joint velocity error
    double sumsquare_diffvel = 0;
    for (int i=0; i< vecdiffvel.size(); ++i) {
        sumsquare_diffvel += vecdiffvel[i] * vecdiffvel[i];
        // std::cout << vecdiffvel[i] << ' ';
    }
    // std::cout << std::endl;

    /* In original deepmimic
	const double vel_scale = 0.1 / 15 * num_joints;
	const double err_scale = 1;
    double vel_reward = exp(-err_scale * vel_scale * vel_err);
    */

    // int num_joints = 10 + 6; // revolute (10), spherical (6)
    // const double vel_scale = 0.001 / 15 * num_joints;
    // const double err_scale = 1;
    // double vel_reward = exp(-err_scale * vel_scale * sumdiffvel);

    /*
    if (std::isnan(vel_reward)) {
        std::cout << "vel_reward is nan!" << std::endl;
        //assert(!isnan(vel_reward));
    }
    */

    // return vel_reward;
    return sumsquare_diffvel;
}


double DeepmimicUtility::get_endeffPosReward(int n_endeff, Eigen::MatrixXd &gc_endeff, Eigen::MatrixXd &gc_ref_endeff)
{
    std::vector<double> vecdiffendeff;
    vecdiffendeff.clear();

    double ref_body_y_offset = m_refmotions[m_current_ref_motion_index].get_ref_body_y_offset();

    double squared_distance;
    for (int idx=0; idx<n_endeff; idx++) {
        squared_distance = (gc_endeff(idx, 0) - gc_ref_endeff(idx, 0))*(gc_endeff(idx, 0) - gc_ref_endeff(idx, 0))
                + (gc_endeff(idx, 1) - (gc_ref_endeff(idx, 1)+ref_body_y_offset))*(gc_endeff(idx, 1) - (gc_ref_endeff(idx, 1)+ref_body_y_offset))
                + (gc_endeff(idx, 2) - gc_ref_endeff(idx, 2))*(gc_endeff(idx, 2) - gc_ref_endeff(idx, 2));
        vecdiffendeff.push_back(sqrt(squared_distance));
    }

    double sumsquare_diffendeff = 0;
    for (int i=0; i< vecdiffendeff.size(); ++i) {
        sumsquare_diffendeff += vecdiffendeff[i] * vecdiffendeff[i];
        /*
        if (std::isnan(vecdiffendeff[i]) || std::isnan(sumdiffendeff)) {
            std::cout << "vecdiffendeff[" << i << "] :" << vecdiffendeff[i] << ' ' << std::endl;
            std::cout << "gc_endeff(,0) : " << gc_endeff(i, 0) << ", gc_ref_endeff(,0) : " << gc_ref_endeff(i, 0) << std::endl;
        }
        */
    }

    /* In original deepmimic
    const double end_eff_scale = 10;
    const double err_scale = 1;
    double curr_end_err = (pos_rel1 - pos_rel0).squaredNorm();
	end_eff_err += curr_end_err;
    double end_eff_reward = exp(-err_scale * end_eff_scale * end_eff_err);
    */

    // const double end_eff_scale = 10;
    // const double err_scale = 1;
    // double end_eff_reward = exp(-err_scale * end_eff_scale * sumdiffendeff);

    /*
    if (std::isnan(end_eff_reward)) {
        std::cout << "end_eff_reward is nan!" << std::endl;
        std::cout << "sumdiffendeff : " << sumdiffendeff << std::endl;
    }
    */

    // return end_eff_reward;
    return sumsquare_diffendeff;
}


double DeepmimicUtility::get_comPosReward(Eigen::VectorXd &vcom1, Eigen::VectorXd &vcom2)
{
    double squared_distance, height_COM, pel_vel;
    // double ref_body_y_offset = m_refmotions[m_current_ref_motion_index].get_ref_body_y_offset();
    // squared_distance = (vcom1(0) - vcom2(0))*(vcom1(0) - vcom2(0))
    //         + (vcom1(1) - (vcom2(1)+ref_body_y_offset))*(vcom1(1) - (vcom2(1)+ref_body_y_offset))
    //         + (vcom1(2) - vcom2(2))*(vcom1(2) - vcom2(2));

    /// CoM height and pelvis velocity in pelvis frame, yjkoo 20210518
    height_COM = (vcom1(2) - vcom2(2))*(vcom1(2) - vcom2(2));
    pel_vel =   (vcom1(3) - vcom2(3))*(vcom1(3) - vcom2(3))
              + (vcom1(4) - vcom2(4))*(vcom1(4) - vcom2(4))
              + (vcom1(5) - vcom2(5))*(vcom1(5) - vcom2(5));

    squared_distance = 10.*height_COM + pel_vel;

    /* In original deepmimic
    const double com_scale = 10;
    const double err_scale = 1;
    com_err = 0.1 * (com_vel1_world - com_vel0_world).squaredNorm();
    double com_reward = exp(-err_scale * com_scale * com_err);
     */

    // const double com_scale = 10;
    // const double err_scale = 1;
    // double com_err = 0.1 * sqrt(squared_distance);
    // double com_reward = exp(-err_scale * com_scale * squared_distance);

    // return com_reward;
    return squared_distance;
}

/*
double DeepmimicUtility::get_rootReward()
{
    double root_err = m_root_pos_err
               + 0.1 * m_root_rot_err
               + 0.01 * m_root_vel_err
               + 0.001 * m_root_ang_vel_err;

    // const double root_scale = 5;
    // const double err_scale = 1;
    // double root_reward = exp(-err_scale * root_scale * root_err);


    // if (std::isnan(root_reward)) {
    //     std::cout << "root_reward is nan!" << std::endl;
    // }


    // return root_reward;
    return root_err;
}
*/

#endif //_RAISIM_GYM_DEEPMIMICUTILITY_H

//
// Created by skoo on 20. 7. 2..
//

#ifndef _RAISIM_GYM_DEEPMIMICUTILITY_H
#define _RAISIM_GYM_DEEPMIMICUTILITY_H

#define NROBOTDOF 37   // <<---- HARD CODING

#include <cmath>
#include <random>
#include <nlohmann/json.hpp>
#include "quaternion_mskbiodyn.hpp"


class RefMotionStorage {
public:
    RefMotionStorage() {}
    ~RefMotionStorage() {}

public:
    double m_ref_motion_2array[500][NROBOTDOF];
    double m_cumulframetime_1array[500];
    int m_nframe = 0;
    double m_totalframetime = 0;
    bool m_bcyclic = false;
};


class DeepmimicUtility {

public:
    DeepmimicUtility() {};
    ~DeepmimicUtility() {};

    void read_ref_motion(std::string filename);
    void set_ref_motion_lateral_offset(double body_y_offset);
    void get_gc_ref_motion(Eigen::VectorXd &ref_motion_one, double tau_in);
    static void get_gv_ref_motion(Eigen::VectorXd &gv, Eigen::VectorXd &gc1, Eigen::VectorXd &gc2, double dt);
    double get_phase(double tau_in);
    double get_tau_random_start();
    double get_duration_ref_motion() {return m_totalframetime;}
    bool iscyclic() {return m_bcyclic;}

    double get_angularPosReward(Eigen::VectorXd &gc, Eigen::VectorXd &gc_ref);
    double get_angularVelReward(Eigen::VectorXd &gv, Eigen::VectorXd &gv_ref);
    double get_endeffPosReward(Eigen::MatrixXd &gc_endeff, Eigen::MatrixXd &gc_ref_endeff);
    double get_comPosReward(Eigen::VectorXd &vcom1, Eigen::VectorXd &vcom2);
    double get_rootReward();

private:
    void calc_slerp_joint(Eigen::VectorXd &, const int *, int, int, double);
    void calc_interp_joint(Eigen::VectorXd &, int , int, int, double); // static void slerp(Quaternion &qout, Quaternion &q1, Quaternion &q2, double lambda);
    static void calc_rotvel_joint(Eigen::VectorXd &gv, const int *idx_gv, Eigen::VectorXd &gc1, Eigen::VectorXd &gc2, const int *idx_gc, double dt);
    static void calc_linvel_joint(Eigen::VectorXd &gv, int idx_gv, Eigen::VectorXd &gc1, Eigen::VectorXd &gc2, int idx_gc, double dt);
    void get_cycle_tau(int &ncycle, double &tau, double tau_in);

private:
    std::string m_ref_motion_file;
    std::vector<Eigen::VectorXd> m_ref_motion;
    std::vector<double> m_cumulframetime;
    int m_nframe = 0;
    double m_totalframetime = 0;
    bool m_bcyclic = false;
    double m_root_pos_err, m_root_rot_err, m_root_vel_err, m_root_ang_vel_err;
    double m_ref_body_y_offset = 0;
    // RefMotionStorage m_refmotionstorage;
};


void DeepmimicUtility::set_ref_motion_lateral_offset(double body_y_offset)
{
    for (int iframe = 0; iframe < m_nframe; iframe++) {
        m_ref_motion[iframe](1) -= m_ref_body_y_offset;
    }

    m_ref_body_y_offset = body_y_offset;
}


void DeepmimicUtility::read_ref_motion(std::string filename) {
    // read motion data from a reference file
    m_ref_motion_file = std::move(filename);
    struct stat buffer{};

    std::ifstream infile1(m_ref_motion_file);
    // std::cout << m_ref_motion_file << std::endl;
    nlohmann::json jsondata;
    infile1 >> jsondata;

    // 20200706 SKOO check if the reference motion is cyclic (wrap) or not
    std::string strloop = jsondata["Loop"];
    std::transform(strloop.begin(), strloop.end(), strloop.begin(), ::tolower);

    if (strloop == "wrap")
        m_bcyclic = true;

    // joint index conversion between raisim model and motion data
    Eigen::VectorXd jointIdxConvTable(NROBOTDOF);
    jointIdxConvTable <<
        1, 3, 2,         // gc_root[3]
        4, 5, 6, 7,      // gc_qroot[4]
        8, 9, 10, 11,    // qlumbar[4]
        20, 21, 22, 23,  // qrshoulder[4]
        24,              // relbow
        33, 34, 35, 36,  // qlshoulder[4]
        37,              // lelbow
        12, 13, 14, 15,  // qrhip[4]
        16,              // rknee
        17, 18, 19,      // qrankle[3]
        25, 26, 27, 28,  // qlhip[4]
        29,              // lknee
        30, 31, 32;      // qlankle[3]

    Eigen::VectorXd jointNominalConfig(NROBOTDOF);

    int nframe = jsondata["Frames"].size();
    double totaltime = 0;
    // m_ref_motion.clear();
    // m_cumulframetime.clear();
    m_ref_motion.resize(nframe);
    m_cumulframetime.resize(nframe);

    int ngcjoint = NROBOTDOF;

    for (int iframe = 0; iframe < nframe; iframe++) {
        jointNominalConfig.setZero();
        for (int igcjoint = 0; igcjoint < ngcjoint; igcjoint++)
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


void DeepmimicUtility::get_gv_ref_motion(
        Eigen::VectorXd &gv,
        Eigen::VectorXd &gc1,
        Eigen::VectorXd &gc2,
        double dt)
{
    int idx_gc_root[3]       = {0, 1, 2};
    int idx_gc_qroot[4]      = {3, 4, 5, 6};
    int idx_gc_qlumbar[4]    = {7, 8, 9, 10};
    int idx_gc_qrshoulder[4] = {11, 12, 13, 14};
    int idx_gc_relbow        = 15;
    int idx_gc_qlshoulder[4] = {16, 17, 18, 19};
    int idx_gc_lelbow        = 20;
    int idx_gc_qrhip[4]      = {21, 22, 23, 24};
    int idx_gc_rknee         = 25;
    int idx_gc_rankle        = 26;
    int idx_gc_rsubtalar     = 27;
    int idx_gc_rmtp          = 28;
    int idx_gc_qlhip[4]      = {29, 30, 31, 32};
    int idx_gc_lknee         = 33;
    int idx_gc_lankle        = 34;
    int idx_gc_lsubtalar     = 35;
    int idx_gc_lmtp          = 36;

    int idx_gv_root[3]       = {0, 1, 2};
    int idx_gv_qroot[3]      = {3, 4, 5};
    int idx_gv_qlumbar[3]    = {6, 7, 8};
    int idx_gv_qrshoulder[3] = {9, 10, 11};
    int idx_gv_relbow        = 12;
    int idx_gv_qlshoulder[3] = {13, 14, 15};
    int idx_gv_lelbow        = 16;
    int idx_gv_qrhip[3]      = {17, 18, 19};
    int idx_gv_rknee         = 20;
    int idx_gv_rankle        = 21;
    int idx_gv_rsubtalar     = 21;
    int idx_gv_rmtp          = 21;
    int idx_gv_qlhip[3]      = {24, 25, 26};
    int idx_gv_lknee         = 27;
    int idx_gv_lankle        = 28;
    int idx_gv_lsubtalar     = 29;
    int idx_gv_lmtp          = 30;

    calc_rotvel_joint(gv, idx_gv_qroot,      gc1, gc2, idx_gc_qroot, dt);
    calc_rotvel_joint(gv, idx_gv_qlumbar,    gc1, gc2, idx_gc_qlumbar, dt);
    calc_rotvel_joint(gv, idx_gv_qrshoulder, gc1, gc2, idx_gc_qrshoulder, dt);
    calc_rotvel_joint(gv, idx_gv_qlshoulder, gc1, gc2, idx_gc_qlshoulder, dt);
    calc_rotvel_joint(gv, idx_gv_qrhip,      gc1, gc2, idx_gc_qrhip, dt);
    calc_rotvel_joint(gv, idx_gv_qlhip,      gc1, gc2, idx_gc_qlhip, dt);

    calc_linvel_joint(gv, idx_gv_root[0],    gc1, gc2, idx_gc_root[0], dt);
    calc_linvel_joint(gv, idx_gv_root[1],    gc1, gc2, idx_gc_root[1], dt);
    calc_linvel_joint(gv, idx_gv_root[2],    gc1, gc2, idx_gc_root[2], dt);
    calc_linvel_joint(gv, idx_gv_relbow,     gc1, gc2, idx_gc_relbow, dt);
    calc_linvel_joint(gv, idx_gv_lelbow,     gc1, gc2, idx_gc_lelbow, dt);
    calc_linvel_joint(gv, idx_gv_rknee,      gc1, gc2, idx_gc_rknee, dt);
    calc_linvel_joint(gv, idx_gv_rankle,     gc1, gc2, idx_gc_rankle, dt);
    calc_linvel_joint(gv, idx_gv_rsubtalar,  gc1, gc2, idx_gc_rsubtalar, dt);
    calc_linvel_joint(gv, idx_gv_rmtp,       gc1, gc2, idx_gc_rmtp, dt);
    calc_linvel_joint(gv, idx_gv_lknee,      gc1, gc2, idx_gc_lknee, dt);
    calc_linvel_joint(gv, idx_gv_lankle,     gc1, gc2, idx_gc_lankle, dt);
    calc_linvel_joint(gv, idx_gv_lsubtalar,  gc1, gc2, idx_gc_lsubtalar, dt);
    calc_linvel_joint(gv, idx_gv_lmtp,       gc1, gc2, idx_gc_lmtp, dt);
}


void DeepmimicUtility::calc_linvel_joint(
        Eigen::VectorXd &gv,    // output angular velocity
        const int idx_gv,
        Eigen::VectorXd &gc1,
        Eigen::VectorXd &gc2,
        const int idx_gc,
        double dt)
{
    gv(idx_gv) = (gc2(idx_gc) - gc1(idx_gc)) / dt;
}


void DeepmimicUtility::calc_rotvel_joint(
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


double DeepmimicUtility::get_phase(double tau_in)
{
    int ncycle;
    double tau;

    get_cycle_tau(ncycle, tau, tau_in);

    double phase = tau/m_totalframetime; // [0, 1]

    return phase;
}


double DeepmimicUtility::get_tau_random_start()
{
    std::random_device rd;  // random device
    std::mt19937 mersenne(rd()); // random generator, a mersenne twister
    std::uniform_real_distribution<double> distribution(0.0,1.0);
    double random_number = distribution(mersenne);
    double tau_random_start = 0;

    if (iscyclic())
        tau_random_start = get_duration_ref_motion() * 0.9 * random_number;
    else
        tau_random_start = get_duration_ref_motion() * 0.5 * random_number;

    return tau_random_start;
}


void DeepmimicUtility::get_cycle_tau(int &ncycle, double &tau, double tau_in)
{
    ncycle = 0;
    tau = tau_in;

    // In case that the reference motion is cyclic
    if (m_bcyclic) {
        if (tau_in > m_totalframetime) {  // SKOO 20200706 check when equal?
            ncycle = (int) (tau_in / m_totalframetime);
            tau = tau_in - m_totalframetime * ncycle;
        }
    } else {
        // assert(tau < m_totalframetime); // SKOO 20200706 check when equal?
        if (tau > m_totalframetime)
            std::cout << "Error : ref motion time over" << std::endl;
    }
    // std::cout << m_totalframetime << " : " << tau_in << " : " << ncycle << " : " << tau << std::endl;
}


void DeepmimicUtility::get_gc_ref_motion(Eigen::VectorXd &ref_motion_one, double tau_in)
{
    int ncycle;
    double tau;

    get_cycle_tau(ncycle, tau, tau_in);

    int idx_frame_prev = -1;
    int idx_frame_next = -1;
    double t_offset_base;
    double t_interval, t_interp;

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

    int idx_root[3]       = {0, 1, 2};
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

    calc_slerp_joint(ref_motion_one, idx_qroot, idx_frame_prev, idx_frame_next, t_interp);
    calc_slerp_joint(ref_motion_one, idx_qlumbar, idx_frame_prev, idx_frame_next, t_interp);
    calc_slerp_joint(ref_motion_one, idx_qrshoulder, idx_frame_prev, idx_frame_next, t_interp);
    calc_slerp_joint(ref_motion_one, idx_qlshoulder, idx_frame_prev, idx_frame_next, t_interp);
    calc_slerp_joint(ref_motion_one, idx_qrhip, idx_frame_prev, idx_frame_next, t_interp);
    calc_slerp_joint(ref_motion_one, idx_qlhip, idx_frame_prev, idx_frame_next, t_interp);

    calc_interp_joint(ref_motion_one, idx_root[0],   idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_root[1],   idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_root[2],   idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_relbow,    idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_lelbow,    idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_rknee,     idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_rankle,    idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_rsubtalar, idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_rmtp,      idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_lknee,     idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_lankle,    idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_lsubtalar, idx_frame_prev, idx_frame_next, t_interp);
    calc_interp_joint(ref_motion_one, idx_lmtp,      idx_frame_prev, idx_frame_next, t_interp);

    if (m_bcyclic) {
        double xgap   = (m_ref_motion[m_nframe - 1][0] - m_ref_motion[0][0]) / (m_nframe - 1);
        double xcycle = (m_ref_motion[m_nframe - 1][0] - m_ref_motion[0][0]) + xgap;
        ref_motion_one[0] += xcycle * ncycle; // 20200706 SKOO check it again?
    }
}


void DeepmimicUtility::calc_slerp_joint(
        Eigen::VectorXd &ref_motion_one,                // output
        const int *idxlist,                             // indices to be updated
        int idxprev,                                    // input
        int idxnext,                                    // input
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


void DeepmimicUtility::calc_interp_joint(
        Eigen::VectorXd &ref_motion_one,                // output
        int idxjoint,                                   // index of a scalar joint to be updated
        int idxprev,                                    // input
        int idxnext,                                    // input
        double t_interp) {                        // input

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


double DeepmimicUtility::get_angularPosReward(Eigen::VectorXd &gc, Eigen::VectorXd &gc_ref)
{
    int idx_root[3]       = {0, 1, 2};
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

    // SKOO 20200708 should we include the base orientation? Not sure...
    // q1.set(gc(idx_qroot[0]), gc(idx_qroot[1]), gc(idx_qroot[2]), gc(idx_qroot[3]));
    // q2.set(gc_ref(idx_qroot[0]), gc_ref(idx_qroot[1]), gc_ref(idx_qroot[2]), gc_ref(idx_qroot[3]));
    // vecdiffpos.push_back(Quaternion::get_angle_two_quaternions(q1, q2));

    q1.set(gc(idx_qlumbar[0]), gc(idx_qlumbar[1]), gc(idx_qlumbar[2]), gc(idx_qlumbar[3]));
    q2.set(gc_ref(idx_qlumbar[0]), gc_ref(idx_qlumbar[1]), gc_ref(idx_qlumbar[2]), gc_ref(idx_qlumbar[3]));
    vecdiffpos.push_back(Quaternion::get_angle_two_quaternions(q1, q2));

    q1.set(gc(idx_qrshoulder[0]), gc(idx_qrshoulder[1]), gc(idx_qrshoulder[2]), gc(idx_qrshoulder[3]));
    q2.set(gc_ref(idx_qrshoulder[0]), gc_ref(idx_qrshoulder[1]), gc_ref(idx_qrshoulder[2]), gc_ref(idx_qrshoulder[3]));
    vecdiffpos.push_back(Quaternion::get_angle_two_quaternions(q1, q2));

    q1.set(gc(idx_qlshoulder[0]), gc(idx_qlshoulder[1]), gc(idx_qlshoulder[2]), gc(idx_qlshoulder[3]));
    q2.set(gc_ref(idx_qlshoulder[0]), gc_ref(idx_qlshoulder[1]), gc_ref(idx_qlshoulder[2]), gc_ref(idx_qlshoulder[3]));
    vecdiffpos.push_back(Quaternion::get_angle_two_quaternions(q1, q2));

    q1.set(gc(idx_qrhip[0]), gc(idx_qrhip[1]), gc(idx_qrhip[2]), gc(idx_qrhip[3]));
    q2.set(gc_ref(idx_qrhip[0]), gc_ref(idx_qrhip[1]), gc_ref(idx_qrhip[2]), gc_ref(idx_qrhip[3]));
    vecdiffpos.push_back(Quaternion::get_angle_two_quaternions(q1, q2));

    q1.set(gc(idx_qlhip[0]), gc(idx_qlhip[1]), gc(idx_qlhip[2]), gc(idx_qlhip[3]));
    q2.set(gc_ref(idx_qlhip[0]), gc_ref(idx_qlhip[1]), gc_ref(idx_qlhip[2]), gc_ref(idx_qlhip[3]));
    vecdiffpos.push_back(Quaternion::get_angle_two_quaternions(q1, q2));

    // vecdiffpos.push_back(abs(gc(idx_root[0])  - gc_ref(idx_root[0])));
    // vecdiffpos.push_back(abs(gc(idx_root[1])  - (gc_ref(idx_root[1])+m_ref_body_y_offset)));
    // vecdiffpos.push_back(abs(gc(idx_root[2])  - gc_ref(idx_root[2])));
    vecdiffpos.push_back(abs(gc(idx_relbow)    - gc_ref(idx_relbow)));
    vecdiffpos.push_back(abs(gc(idx_lelbow)    - gc_ref(idx_lelbow)));
    vecdiffpos.push_back(abs(gc(idx_rknee)     - gc_ref(idx_rknee)));
    vecdiffpos.push_back(abs(gc(idx_rankle)    - gc_ref(idx_rankle)));
    vecdiffpos.push_back(abs(gc(idx_rsubtalar) - gc_ref(idx_rsubtalar)));
    vecdiffpos.push_back(abs(gc(idx_rmtp)      - gc_ref(idx_rmtp)));
    vecdiffpos.push_back(abs(gc(idx_lknee)     - gc_ref(idx_lknee)));
    vecdiffpos.push_back(abs(gc(idx_lankle)    - gc_ref(idx_lankle)));
    vecdiffpos.push_back(abs(gc(idx_lsubtalar) - gc_ref(idx_lsubtalar)));
    vecdiffpos.push_back(abs(gc(idx_lmtp)      - gc_ref(idx_lmtp)));

    q1.set(gc(idx_qroot[0]), gc(idx_qroot[1]), gc(idx_qroot[2]), gc(idx_qroot[3]));
    q2.set(gc_ref(idx_qroot[0]), gc_ref(idx_qroot[1]), gc_ref(idx_qroot[2]), gc_ref(idx_qroot[3]));
    m_root_rot_err = Quaternion::get_angle_two_quaternions(q1, q2);

    m_root_pos_err = sqrt(
            (gc(idx_root[0])  - gc_ref(idx_root[0])) * (gc(idx_root[0])  - gc_ref(idx_root[0]))
            + (gc(idx_root[1])  - (gc_ref(idx_root[1])+m_ref_body_y_offset)) * (gc(idx_root[1])  - (gc_ref(idx_root[1])+m_ref_body_y_offset))
            + (gc(idx_root[2])  - gc_ref(idx_root[2])) * (gc(idx_root[2])  - gc_ref(idx_root[2])));

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

    /*
    int idx_gv_qlumbar[3]    = {6, 7, 8};
    int idx_gv_qrshoulder[3] = {9, 10, 11};
    int idx_gv_relbow        = 12;
    int idx_gv_qlshoulder[3] = {13, 14, 15};
    int idx_gv_lelbow        = 16;
    int idx_gv_qrhip[3]      = {17, 18, 19};
    int idx_gv_rknee         = 20;
    int idx_gv_rankle        = 21;
    int idx_gv_rsubtalar     = 22;
    int idx_gv_rmtp          = 23;
    int idx_gv_qlhip[3]      = {24, 25, 26};
    int idx_gv_lknee         = 27;
    int idx_gv_lankle        = 28;
    int idx_gv_lsubtalar     = 29;
    int idx_gv_lmtp          = 30;
    */

    // for (int i=0; i< 34; ++i) std::cout << gv(i) << ' '; std::cout << std::endl;
    // for (int i=0; i< 34; ++i) std::cout << gv_ref(i) << ' '; std::cout << std::endl;

    for (int idx=3; idx<31; idx++) {
        vecdiffvel.push_back((gv(idx) - gv_ref(idx)));
    }

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


double DeepmimicUtility::get_endeffPosReward(Eigen::MatrixXd &gc_endeff, Eigen::MatrixXd &gc_ref_endeff)
{
    std::vector<double> vecdiffendeff;
    vecdiffendeff.clear();

    int n_endeff = 4; // two hands and two feet
    double squared_distance;
    for (int idx=0; idx<n_endeff; idx++) {
        squared_distance = (gc_endeff(idx, 0) - gc_ref_endeff(idx, 0))*(gc_endeff(idx, 0) - gc_ref_endeff(idx, 0))
                + (gc_endeff(idx, 1) - (gc_ref_endeff(idx, 1)+m_ref_body_y_offset))*(gc_endeff(idx, 1) - (gc_ref_endeff(idx, 1)+m_ref_body_y_offset))
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
    double squared_distance;
    squared_distance = (vcom1(0) - vcom2(0))*(vcom1(0) - vcom2(0))
            + (vcom1(1) - (vcom2(1)+m_ref_body_y_offset))*(vcom1(1) - (vcom2(1)+m_ref_body_y_offset))
            + (vcom1(2) - vcom2(2))*(vcom1(2) - vcom2(2));
    // std::cout << squared_distance << std::endl;

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

    /*
    if (std::isnan(com_reward)) {
        std::cout << "com_reward is nan!" << std::endl;
        std::cout << "vcom1(0) : " << vcom1(0) << ", vcom2(0) : " << vcom2 << std::endl;
    }
    */

    // return com_reward;
    return squared_distance;
}


double DeepmimicUtility::get_rootReward()
{
    double root_err = m_root_pos_err
               + 0.1 * m_root_rot_err
               + 0.01 * m_root_vel_err
               + 0.001 * m_root_ang_vel_err;

    // const double root_scale = 5;
    // const double err_scale = 1;
    // double root_reward = exp(-err_scale * root_scale * root_err);

    /*
    if (std::isnan(root_reward)) {
        std::cout << "root_reward is nan!" << std::endl;
    }
    */

    // return root_reward;
    return root_err;
}


#endif //_RAISIM_GYM_DEEPMIMICUTILITY_H

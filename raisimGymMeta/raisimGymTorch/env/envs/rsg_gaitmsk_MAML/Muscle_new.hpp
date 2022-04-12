#include <iostream>
#include <cmath>
#include <nlohmann/json.hpp>
#include <raisim/World.hpp>
#include <raisim/RaisimServer.hpp>

class MuscleModel {
public:
    raisim::PolyLine *Path;
    raisim::ArticulatedSystem *model;

    std::vector<std::string> m_strTargetJoint; // [3]
    int m_nTargetPath[3][2];

    // for conditional path
    std::vector<std::string> Coordinate; //[9]
    std::vector<std::string> Range; //[9]
    std::vector<int> isRange; // [9]

    // for moving path
    std::vector<std::string> CoordinateX, CoordinateY, CoordinateZ; //[9]
    std::vector<std::string> x_x, x_y, y_x, y_y, z_x, z_y; //[9]

    // Muscle constants
    std::vector<double> vx, vy, vx2, vy2;
    std::vector<double> x_b, x_c, x_d;
    std::vector<double> y_b, y_c, y_d;

    // muscle dynamics functions
    MuscleModel();
    static void calcCoefficients(const std::vector<double> &x, const std::vector<double> &y, std::vector<double> &b, std::vector<double> &c, std::vector<double> &d);
    void UpdateGlobalPos(const std::vector<raisim::Vec<3>> &trs, const std::vector<raisim::Mat<3, 3>> &rot);
    void CalcTotalLength();
    double CalcPassiveTendonForce(double dTendonLength) const;
    double CalcPassiveFiberForce(double dFiberLength) const;
    double CalcActiveFiberForce(double dFiberLength, double dFiberVelocity);
    void CalcTendonFiberLength();
    void Update(const std::vector<raisim::Vec<3>> &trs, const std::vector<raisim::Mat<3, 3>> &rot, std::vector<double> &ForceTable);
    void CalcMomentArm();
    void ApplyForce(std::vector<double> &ForceTable);
    void VisOn(raisim::RaisimServer &Server);
    void VisUpdate(raisim::RaisimServer &Server);
    void SetScaleLength(double scale);
    void SetScaleMuscle(const std::vector<std::string> &scale_link, const std::vector<double> &scale_value, const std::vector<raisim::Vec<3>> &trs, const std::vector<raisim::Mat<3, 3>> &rot, double l0);
    void initialize(const std::vector<raisim::Vec<3>> &trs, const std::vector<raisim::Mat<3, 3>> &rot);
    void CalcActivation();
    double CalcPennationAngle(double tendon_length);
    static raisim::Vec<3> mycross(const raisim::Vec<3> &A, const raisim::Vec<3> &B);
    static double mydot(const raisim::Vec<3> &A, const raisim::Vec<3> &B);
    static double quat2angle(const double (&q1)[4]);
    void SetPennationAngle(double dPennationAngle) {m_const_OptPennationAngle = dPennationAngle; }
    void SetTendonSlackLength(double dTendonSlackLength) {m_constTendonSlackLength = dTendonSlackLength; }
    void SetOptFiberLength(double dOptFiberLength) {m_const_OptFiberLength = dOptFiberLength; }
    void SetFiso(double dFiso) {m_const_Fiso = dFiso; }
    void SetControlDt(double dt) {m_const_control_dt = dt; }
    void SetHillTypeOn() {m_HillTypeOn = 1; }
    void SetHillTypeOff() {m_HillTypeOn = 0; }
    double GetFiso() {return m_const_Fiso; }
    void SetActivation(double dActivation) {m_muscleActivation = dActivation; }
    void SetExcitation(double dExcitation) {m_muscleExcitation = dExcitation; }
    void SetMuscleName(std::string strMusclename) {m_strMuscleName = strMusclename; }
    void SetNumPathPoint(std::size_t nPathPoint) {m_nPathPoint = nPathPoint; }
    void SetNumTargetJoint(std::size_t nTargetJoint) {m_nTargetJoint = nTargetJoint; }
    std::size_t GetNumTargetJoint() {return m_nTargetJoint; }
    std::string GetMuscleName() {return m_strMuscleName; }
    void SetPathPosLocal(std::size_t idx, double x, double y, double z) {m_PathPosLocal[idx][0] = x; m_PathPosLocal[idx][1] = y; m_PathPosLocal[idx][2] = z; }
    void SetPathLink(std::size_t idx, std::string pathlink) {m_PathLink[idx] = pathlink; }
    void SetPathType(std::size_t idx, std::string pathtype) {m_PathType[idx] = pathtype; }
    std::string GetPathType(std::size_t idx) {return m_PathType[idx]; }

private:
    std::string m_strMuscleName;
    int m_HillTypeOn = 1;
    // Muscle parameters for passive fiber force using force-m_muscleLength relationship
    double m_const_KShapeFLCurve = 4.0;
    double m_const_FiberStrainAtFmax = 0.6;

    // Muscle parameters for passive tendon force
    double m_const_TendonStrainAtFmax = 0.033;
    double m_const_ktoe = 3.0; // shape factor for tendon force
    double m_const_Ftoe = 0.33; // see the tendon force equation
    double m_const_etoe = -1.0; // it needs initialization
    double m_const_klin = -1.0; // it needs initialization

    // Muscle parameters for active fiber force
    double m_const_vmax = 10.0;
    double m_const_Af = 0.3;
    double m_const_Flen = 1.8;
    double m_const_gamma = 0.4;

    // Muscle parameters
    double m_const_OptPennationAngle = 0.0; // It needs check.
    double m_const_muscle_height = -1.0;
    double m_constTendonSlackLength = 0.1;
    double m_const_OptFiberLength = 0.1;
    double m_const_Fiso = -1.0;

    // Muscle state
    double m_muscleExcitation = 0.0;
    double m_muscleActivation = -10.0;
    double m_muscleLength = 0.0;
    double InitLength = 0.0;
    double m_currentFiberLength = 0.0;
    double m_currentFiberVelocity = 0.0;
    double m_currentTendonLength = 0.0;

    // Muscle Forces
    double m_TendonForce = 0.0; // passive tendon force
    double m_ActiveFiberForce = 0.0; // active fiber force
    double m_PassiveFiberForce = 0.0; // passive fiber force
    double m_FiberForceAlongTendon = 0.0; // fiber force along tendon
    double m_const_minimum_fiber_length = 0.0;

    // Muscle specific geometry
    int m_nPathPoint = 9;
    int m_nTargetJoint = 0;
    std::vector<raisim::Vec<3>> m_PathPosLocal; // [9]
    std::vector<raisim::Vec<3>> m_PathPosGlobal; // [9]
    std::vector<std::string> m_PathLink; // [9]
    std::vector<std::string> m_PathType; // [9]

    // for moving path
    double fx, dfx, fy, dfy;

    double m_moment_arm[3];
    double m_moment_arm_spherical[3][2];

    // ...
    double m_const_control_dt = 0.01;
};


MuscleModel::MuscleModel() {
    m_strTargetJoint.resize(3);
    m_PathPosLocal.resize(9);
    m_PathPosGlobal.resize(9);
    m_PathLink.resize(9);
    m_PathType.resize(9);

    // for conditional path
    Coordinate.resize(9);
    Range.resize(9);
    isRange.resize(9);

    // for moving path
    CoordinateX.resize(9);
    CoordinateY.resize(9);
    CoordinateZ.resize(9);
    x_x.resize(9);
    x_y.resize(9);
    y_x.resize(9);
    y_y.resize(9);
    z_x.resize(9);
    z_y.resize(9);

    double m_const_etoe = (0.99e2 * m_const_TendonStrainAtFmax * exp(0.3e1)) / (0.166e3 * exp(0.3e1) - 0.67e2);
    double m_const_klin = (0.67e2 / 0.100e3) * 1.0 /
        (m_const_TendonStrainAtFmax - (0.99e2 * m_const_TendonStrainAtFmax * exp(0.3e1)) / (0.166e3 * exp(0.3e1) - 0.67e2));
}

/// cross product
raisim::Vec<3> MuscleModel::mycross(const raisim::Vec<3> &A, const raisim::Vec<3> &B) {
    raisim::Vec<3> C = {
        A[1] * B[2] - A[2] * B[1],
        -A[0] * B[2] + A[2] * B[0],
        A[0] * B[1] - A[1] * B[0] };
    return C;
}

/// dot product
double MuscleModel::mydot(const raisim::Vec<3> &A, const raisim::Vec<3> &B) {
    double C = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    return C;
}

double MuscleModel::quat2angle(const double (&q1)[4]) {
    double qrot;		// rotation angle in radian
    double arot[3], v1; // rotation axis

    qrot = acos(q1[0]) * 2;

    if (qrot < 0.001) {
        v1 = 0;
    }
    else {
        arot[2] = q1[3] / sin(qrot / 2);
        v1 = arot[2] * qrot;
    }
    return v1;
}

/// calculate positions of path points
void MuscleModel::UpdateGlobalPos(const std::vector<raisim::Vec<3>> &trs, const std::vector<raisim::Mat<3, 3>> &rot) {
    raisim::Vec<3> TrsVec;
    raisim::Mat<3, 3> RotMat{};
    for (int i = 0; i < m_nPathPoint; i++) {
        /// get a rotation matrix and translation vector
        std::size_t idx = model->getBodyIdx(m_PathLink[i]);
        TrsVec = trs[idx];
        RotMat = rot[idx];

        if (m_PathType[i] == "Common" || m_PathType[i] == "Moving") {
            /// calculate local position of moving paht point
            if (m_PathType[i] == "Moving") {
                double angle = std::numeric_limits<double>::min();
                double x = std::numeric_limits<double>::min();
                double y = std::numeric_limits<double>::min();
                if (CoordinateX[i] == "walker_knee_angle_r")
                    angle = model->getGeneralizedCoordinate()[25];
                if (CoordinateX[i] == "walker_knee_angle_l")
                    angle = model->getGeneralizedCoordinate()[33];
                assert(angle != std::numeric_limits<double>::min());

                for (int j = 0; j < vx.size()-1; j++) {  // SKOO 20210507 HERE
                    if ((vx[j] - angle) * (vx[j + 1] - angle) <= 0) {  // SKOO 20210507 HERE
                        double diff = (angle - vx[j]);
                        x = vy[j] + diff * (x_b.at(j) + diff * (x_c.at(j) + diff * x_d.at(j)));
                        fx = x;
                        dfx = -diff * (3 * x_d.at(j) * -diff - 2 * x_c.at(j)) + x_b.at(j);
                        break;
                    }
                    if ((j == vx.size() - 1) && (vx[j] - angle) * (vx[j + 1] - angle) > 0) {
                        x = vy[0];
                    }
                }

                for (int j = 0; j < vx2.size()-1; j++) {  // SKOO 20210507 HERE
                    if ((vx2[j] - angle) * (vx2[j + 1] - angle) <= 0) {  // SKOO 20210507 HERE
                        double diff = (angle - vx2[j]);
                        y = vy2[j] + diff * (y_b.at(j) + diff * (y_c.at(j) + diff * y_d.at(j)));
                        fy = y;
                        dfy = -diff * (3 * y_d.at(j) * -diff - 2 * y_c.at(j)) + y_b.at(j);
                        break;
                    }
                    if ((j == vx2.size() - 1) && (vx2[j] - angle) * (vx2[j + 1] - angle) > 0) {
                        y = vy2[0];
                    }
                }

                assert(x != std::numeric_limits<double>::min());
                assert(y != std::numeric_limits<double>::min());
                m_PathPosLocal[i] = { x, y, m_PathPosLocal[i][2]};
            }
            /// calculate global path point of common and moving path point
            m_PathPosGlobal[i] = {
                        RotMat[0] * m_PathPosLocal[i][0] + RotMat[3] * m_PathPosLocal[i][1] + RotMat[6] * m_PathPosLocal[i][2] + TrsVec[0],
                        RotMat[1] * m_PathPosLocal[i][0] + RotMat[4] * m_PathPosLocal[i][1] + RotMat[7] * m_PathPosLocal[i][2] + TrsVec[1],
                        RotMat[2] * m_PathPosLocal[i][0] + RotMat[5] * m_PathPosLocal[i][1] + RotMat[8] * m_PathPosLocal[i][2] + TrsVec[2] };
        }

        if (m_PathType[i] == "Conditional") {
            std::string::size_type sz;
            double min1 = std::stod(Range[i], &sz);
            double max1 = std::stod(Range[i].substr(sz));
            double angle1;
            if (Coordinate[i] == "hip_flexion_r") {
                double q1[4];
                auto gc = model->getGeneralizedCoordinate();
                q1[0] = gc[21];
                q1[1] = gc[22];
                q1[2] = gc[23];
                q1[3] = gc[24];
                angle1 = quat2angle(q1);
            }
            if (Coordinate[i] == "hip_flexion_l") {
                double q1[4];
                auto gc = model->getGeneralizedCoordinate();
                q1[0] = gc[29];
                q1[1] = gc[30];
                q1[2] = gc[31];
                q1[3] = gc[32];
                angle1 = quat2angle(q1);
            }
            if (Coordinate[i] == "walker_knee_angle_r") {
                angle1 = model->getGeneralizedCoordinate()[25];
            }
            if (Coordinate[i] == "walker_knee_angle_l") {
                angle1 = model->getGeneralizedCoordinate()[33];
            }
            /// check conditional path point
            if (angle1 > min1 && angle1 < max1)
                isRange[i] = 1;
            else
                isRange[i] = 0;
            /// calculate global position of conditional path point in conditional situation
            if (isRange[i] == 1) {
                m_PathPosGlobal[i] = {
                        RotMat[0] * m_PathPosLocal[i][0] + RotMat[3] * m_PathPosLocal[i][1] + RotMat[6] * m_PathPosLocal[i][2] + TrsVec[0],
                        RotMat[1] * m_PathPosLocal[i][0] + RotMat[4] * m_PathPosLocal[i][1] + RotMat[7] * m_PathPosLocal[i][2] + TrsVec[1],
                        RotMat[2] * m_PathPosLocal[i][0] + RotMat[5] * m_PathPosLocal[i][1] + RotMat[8] * m_PathPosLocal[i][2] + TrsVec[2]};
            }
            /// calculate global position of conditional path point in non-conditional situation
            if (isRange[i] == 0) {
                m_PathPosGlobal[i][0] = m_PathPosGlobal[i - 1][0]; assert(i > 0);
                m_PathPosGlobal[i][1] = m_PathPosGlobal[i - 1][1];
                m_PathPosGlobal[i][2] = m_PathPosGlobal[i - 1][2];
            }
        }
    }
}

/// calculate length of total muscle (equal to "fiber_length*cos(pennationa_angle) + tendon length")
void MuscleModel::CalcTotalLength() {
    double length_temp;
    m_muscleLength = 0.0;

    for (int i = 0; i < m_nPathPoint - 1; i++) {
        length_temp = sqrt(pow(m_PathPosGlobal[i][0] - m_PathPosGlobal[i + 1][0], 2)
            + pow(m_PathPosGlobal[i][1] - m_PathPosGlobal[i + 1][1], 2)
            + pow(m_PathPosGlobal[i][2] - m_PathPosGlobal[i + 1][2], 2));
        m_muscleLength += length_temp;
    }
}

/// calculate passive tendon force from tendon length (thelen 2003)
double MuscleModel::CalcPassiveTendonForce(double dTendonLength) const {
    double dTendonStrain = (dTendonLength / m_constTendonSlackLength) - 1.0;
    double dTendonForce = 0.0;
    if (dTendonStrain < 0.0)
        dTendonForce = 0.0;
    else if (dTendonStrain < m_const_etoe)
        dTendonForce = m_const_Ftoe * (exp(m_const_ktoe * dTendonStrain / m_const_etoe) - 1.0) / (exp(m_const_ktoe) - 1.0);
    else
        dTendonForce = m_const_klin * (dTendonStrain - m_const_etoe) + m_const_Ftoe;

    dTendonForce = dTendonForce * m_const_Fiso;
    return dTendonForce;
}

/// calculate passive fiber force from fiber length (thelen 2003)
double MuscleModel::CalcPassiveFiberForce(double dFiberLength) const {
    double dNormFiberLength = dFiberLength / m_const_OptFiberLength;
    double dFiberPassiveForce =
        (exp(m_const_KShapeFLCurve * (dNormFiberLength - 1.) / m_const_FiberStrainAtFmax) - 1.) / (exp(m_const_KShapeFLCurve) - 1.);
    if (dFiberPassiveForce < 0.)
        dFiberPassiveForce = 0.0;

    dFiberPassiveForce = dFiberPassiveForce * m_const_Fiso;
    return dFiberPassiveForce;
}

/// calculate active fiber force from fiber length, fiber velocity and activation (thelen 2003)
double MuscleModel::CalcActiveFiberForce(double dFiberLength, double dFiberVelocity) {
    double dNormFiberLength = dFiberLength / m_const_OptFiberLength;
    double dActiveFiberForce = 0.0;
    double fl = 0.0; // a factor to be multiplied to muscle force

    /// calculate fl (thelen 2003)
    fl = exp(-pow((dNormFiberLength - 1.0), 2.0) / m_const_gamma);
    if (fl < 0.15)
        fl = 0.15;

    /// calculate normalized fiber velocity
    double dNormFiberVelocity = dFiberVelocity / m_const_OptFiberLength;

    /// clip fiber velocity (from OpenSim-Core)
    if (dNormFiberVelocity > 0.8)
        dNormFiberVelocity = 0.8;

    /// calculate active fiber force (thelen 2003)
    double dFiberForceContraction = (-1*(0.25 + 0.75 * m_muscleActivation) * m_const_vmax * m_muscleActivation * fl
        - dNormFiberVelocity * m_muscleActivation * fl) /
        (dNormFiberVelocity / m_const_Af - (0.25 + 0.75 * m_muscleActivation) * m_const_vmax);

    double dFiberForceLengthening = (-1*(2.0 + 2.0/m_const_Af) * m_muscleActivation * fl * m_const_Flen / (m_const_Flen - 1.0) -
        (0.25 + 0.75 * m_muscleActivation) * m_const_vmax * (m_muscleActivation * fl)) /
        (-1*(2.0 + 2.0/m_const_Af) / (m_const_Flen - 1.0) * dNormFiberVelocity - (0.25 + 0.75 * m_muscleActivation) * m_const_vmax);

    if (dFiberForceContraction < fl * m_muscleActivation)
        dActiveFiberForce = dFiberForceContraction;
    else if (dFiberForceLengthening > fl * m_muscleActivation)
        dActiveFiberForce = dFiberForceLengthening;
    else
        dActiveFiberForce = dFiberForceContraction;

    if (dActiveFiberForce < 0.0)
        dActiveFiberForce = 0.0;

    if (dActiveFiberForce > 1.45*m_muscleActivation*fl)
        dActiveFiberForce = dFiberForceContraction;

    dActiveFiberForce = dActiveFiberForce * m_const_Fiso;
    return dActiveFiberForce;
}

/// find force equillibrium state between tendon and fiber (fiber_force*cos(pennation_angle) = tendon_force)
void MuscleModel::CalcTendonFiberLength() {
    /// optimization using bisection method with maximum 20 iterations and force tolerance by 1N

    // calculate initial tendon length, pennation angle, fiber length and fiber velocity
    double dTendonLength = m_constTendonSlackLength + 0.01*m_constTendonSlackLength;
    double dPennationAngle = CalcPennationAngle(dTendonLength);
    double dFiberLength = (m_muscleLength - dTendonLength) / cos(dPennationAngle);
    double dFiberVelocity = (dFiberLength - m_currentFiberLength) / m_const_control_dt;

    // optimization criteria
    const std::size_t Maxiter = 20;
    const double tol = 1.0; // in newton
    double Lower_Boundary = m_constTendonSlackLength;
    double Upper_Boundary = m_muscleLength;

    // It may not happen but in the case that the total muscle length is smaller than the tendon slack length
    if (Upper_Boundary < Lower_Boundary) {
        Lower_Boundary = Upper_Boundary * 0.98;
        dTendonLength = Upper_Boundary * 0.99;
    }

    std::size_t iter = 0;
    double dActiveFiberForce=0.0, dPassiveFiberForce=0.0, dPassiveTendonForce=0.0, ForceDiff=0.0;
    while (true) {
        iter += 1;

        /// calculate the three forces
        dActiveFiberForce = CalcActiveFiberForce(dFiberLength, dFiberVelocity);
        dPassiveFiberForce = CalcPassiveFiberForce(dFiberLength);
        dPassiveTendonForce = CalcPassiveTendonForce(dTendonLength);

        /// update Lt_temp (tendon length) using bisection method
        ForceDiff = (dPassiveTendonForce - (dActiveFiberForce + dPassiveFiberForce) * cos(dPennationAngle));
        if (abs(ForceDiff) < tol || iter > Maxiter) { // update the member variables and exit
            m_currentFiberLength = dFiberLength;
            m_currentTendonLength = dTendonLength;
            m_currentFiberVelocity = dFiberVelocity;
            m_ActiveFiberForce = dActiveFiberForce;
            m_PassiveFiberForce = dPassiveFiberForce;
            m_TendonForce = dPassiveTendonForce;
            m_FiberForceAlongTendon = (dActiveFiberForce + dPassiveFiberForce) * cos(dPennationAngle);
            break;
        }

        if (ForceDiff > 0.0) {
            Upper_Boundary = dTendonLength;
        } else {
            Lower_Boundary = dTendonLength;
        }

        /// update dFiberLength (fiber length) and dFiberVelocity (fiber velocitty) from total muscle length and tendon length
        dTendonLength = (Upper_Boundary + Lower_Boundary) / 2.0;
        dPennationAngle = CalcPennationAngle(dTendonLength);
        dFiberLength = (m_muscleLength - dTendonLength) / cos(dPennationAngle);
        dFiberVelocity = (dFiberLength - m_currentFiberLength) / m_const_control_dt;

        if (dFiberLength < m_const_minimum_fiber_length) {   // minimum fiber length
            dFiberLength = m_const_minimum_fiber_length;
            dFiberVelocity = (dFiberLength - m_currentFiberLength) / m_const_control_dt;
        }
        if (dFiberVelocity < -1*m_const_vmax * m_const_OptFiberLength) { // minimum fiber contraction velocity
            dFiberLength = m_currentFiberLength - m_const_vmax * m_const_OptFiberLength * m_const_control_dt;
            dFiberVelocity = -1*m_const_vmax * m_const_OptFiberLength;
        }
        if (dFiberVelocity > m_const_vmax* m_const_OptFiberLength) {   // maximum fiber contraction velocity
            dFiberLength = m_currentFiberLength + m_const_vmax *m_const_OptFiberLength * m_const_control_dt;
            dFiberVelocity = m_const_vmax *m_const_OptFiberLength;
        }
    }
}


void MuscleModel::Update(const std::vector<raisim::Vec<3>> &trs, const std::vector<raisim::Mat<3, 3>> &rot, std::vector<double> &ForceTable) {
    // update m_PathPosGlobal
    UpdateGlobalPos(trs, rot);

    CalcActivation();

    if (m_HillTypeOn == 1) {
        // update m_muscleLength
        CalcTotalLength();

        // update m_currentFiberLength, m_currentTendonLength, m_currentFiberVelocity
        //    m_ActiveFiberForce, m_PassiveFiberForce, m_TendonForce, m_FiberForceAlongTendon
        CalcTendonFiberLength();
    }
    else {
        m_TendonForce = m_muscleActivation * m_const_Fiso;
    }

    // SKOO 20210626 min and max of m_TendonForce ??? 
  	
	
    /// update m_moment_arm[3] and m_moment_arm_spherical[3][2]
    CalcMomentArm();

    ApplyForce(ForceTable);
}

/// create visual object in RaiSimUnity
void MuscleModel::VisOn(raisim::RaisimServer &Server) {
    Path = Server.addVisualPolyLine(m_strMuscleName);
    for (int iter = 0; iter < m_nPathPoint; iter++) {
        //		Path->points[iter] = {0, 0, 0};
    }
}

/// update visual object in RaiSimUnity
void MuscleModel::VisUpdate(raisim::RaisimServer &Server) {
    Path->points.clear();
    for (int iter = 0; iter < m_nPathPoint; iter++) {
        Path->points.push_back({ m_PathPosGlobal[iter][0], m_PathPosGlobal[iter][1], m_PathPosGlobal[iter][2] });
    }
    Path->color[0] = m_muscleActivation;
    Path->color[1] = 0;
    Path->color[2] = 1 - m_muscleActivation;
}

/// calculate moment arm
void MuscleModel::CalcMomentArm() {
    for (int iter = 0; iter < m_nTargetJoint; iter++) {
        /// conditional path point check
        assert(m_nTargetPath[iter][0] - 1 >= 0);
        if (m_PathType[m_nTargetPath[iter][0] - 1] == "Conditional") {
            if (isRange[m_nTargetPath[iter][0] - 1] == 0) {
                assert(m_nTargetPath[iter][0] - 1 - 1 >= 0);
                if (m_PathType[m_nTargetPath[iter][0] - 1 - 1] == "Conditional") {
                    assert(m_nTargetPath[iter][0] - 1 - 2 >= 0);
                    m_PathPosGlobal[m_nTargetPath[iter][0] - 1] = m_PathPosGlobal[m_nTargetPath[iter][0] - 1 - 2];
                } else {
                    m_PathPosGlobal[m_nTargetPath[iter][0] - 1] = m_PathPosGlobal[m_nTargetPath[iter][0] - 1 - 1];
                }
            }
        }

        assert(m_nTargetPath[iter][1] - 1 >= 0);
        if (m_PathType[m_nTargetPath[iter][1] - 1] == "Conditional") {
            if (isRange[m_nTargetPath[iter][1] - 1] == 0)
                m_PathPosGlobal[m_nTargetPath[iter][1] - 1] = m_PathPosGlobal[m_nTargetPath[iter][1] - 1 + 1];
        }

        m_moment_arm[iter] = 0.;
        m_moment_arm_spherical[iter][0] = 0.;
        m_moment_arm_spherical[iter][1] = 0.;

        if (m_PathType[m_nTargetPath[iter][1] - 1] == "Moving") {
            raisim::Vec<3> TrsVec;
            raisim::Mat<3, 3> RotMat{};
            assert(m_nTargetPath[iter][1] - 1 -1 >= 0);
            std::size_t idx = model->getBodyIdx(m_PathLink[m_nTargetPath[iter][1] - 1 -1]); // femur
            auto str_temp = model->getBodyNames();
            model->getLink(str_temp.at(idx)).getPose(TrsVec, RotMat);

            raisim::Mat<3, 3> IRotMat = RotMat.transpose();
            raisim::Vec<3> ITrsVec = {
                    -IRotMat[0] * TrsVec[0] - IRotMat[3] * TrsVec[1] - IRotMat[6] * TrsVec[2],
                    -IRotMat[1] * TrsVec[0] - IRotMat[4] * TrsVec[1] - IRotMat[7] * TrsVec[2],
                    -IRotMat[2] * TrsVec[0] - IRotMat[5] * TrsVec[1] - IRotMat[8] * TrsVec[2]};

            raisim::Vec<3> pt = m_PathPosGlobal[m_nTargetPath[iter][0] - 1];
            raisim::Vec<3> dt = m_PathPosGlobal[m_nTargetPath[iter][1] - 1];

            raisim::Vec<3> prox = {
            IRotMat[0] * pt[0] + IRotMat[3] * pt[1] + IRotMat[6] * pt[2] + ITrsVec[0],
            IRotMat[1] * pt[0] + IRotMat[4] * pt[1] + IRotMat[7] * pt[2] + ITrsVec[1],
            IRotMat[2] * pt[0] + IRotMat[5] * pt[1] + IRotMat[8] * pt[2] + ITrsVec[2] };
            raisim::Vec<3> dist = {
            IRotMat[0] * dt[0] + IRotMat[3] * dt[1] + IRotMat[6] * dt[2] + ITrsVec[0],
            IRotMat[1] * dt[0] + IRotMat[4] * dt[1] + IRotMat[7] * dt[2] + ITrsVec[1],
            IRotMat[2] * dt[0] + IRotMat[5] * dt[1] + IRotMat[8] * dt[2] + ITrsVec[2] };

            double length = sqrt((prox[0] - dist[0])*(prox[0] - dist[0])
                               + (prox[1] - dist[1])*(prox[1] - dist[1])
                               + (prox[2] - dist[2])*(prox[2] - dist[2]));

            /// dlength/dtheta = {1/2*(length2)^(-1/2)} * {dlength2/dtheta}, length2 = length^2
            /// dlength/dtheta = {        A           } * {       B       }
            ///       B        = 2*(x - x0)*(dx/dtheta) + 2*(y - y0)*(dy/dtheta) + 2*(z - z0)*(dz/dtheta), x: dist[0], x0: prox[0]
            double angle;
            if (m_PathLink[m_nTargetPath[iter][1] - 1] == "tibia_r")
                angle = model->getGeneralizedCoordinate()[25];
            else
                angle = model->getGeneralizedCoordinate()[33];

            double A = 1. / 2. / length;
            double dx = -sin(angle)*fx + cos(angle)*dfx -cos(angle)*fy -sin(angle)*dfy;
            double dy = cos(angle)*fx + sin(angle)*dfx - sin(angle)*fy + cos(angle)*dfy;
            double B = -(2. * (dist[0] - prox[0]) * dx + 2. * (dist[1] - prox[1]) * dy);
            double dlength_dtheta = A * B;
            m_moment_arm[iter] = dlength_dtheta;
        }//
        else {
            raisim::Vec<3> global_joint_pos;
            raisim::ArticulatedSystem::JointRef joint = model->getJoint(m_strTargetJoint[iter]);
            joint.getPosition(global_joint_pos);

            /// a vector from child-path to parent-path
            raisim::Vec<3> line_vec = {
                    m_PathPosGlobal[m_nTargetPath[iter][0] - 1][0] - m_PathPosGlobal[m_nTargetPath[iter][1] - 1][0],
                    m_PathPosGlobal[m_nTargetPath[iter][0] - 1][1] - m_PathPosGlobal[m_nTargetPath[iter][1] - 1][1],
                    m_PathPosGlobal[m_nTargetPath[iter][0] - 1][2] - m_PathPosGlobal[m_nTargetPath[iter][1] - 1][2] };

            /// a vector from child-path to a joint position
            raisim::Vec<3> temp_vec = { global_joint_pos[0] - m_PathPosGlobal[m_nTargetPath[iter][1] - 1][0],
                                       global_joint_pos[1] - m_PathPosGlobal[m_nTargetPath[iter][1] - 1][1],
                                       global_joint_pos[2] - m_PathPosGlobal[m_nTargetPath[iter][1] - 1][2] };

            /// dot product of norm_line_vec and temp_vec to calculate
            /// a perpendicular foot position of the joint_position on the path_line
            line_vec.operator/=(line_vec.norm());   // normalize line vector
            raisim::Vec<3> child_to_foot_vec = line_vec * mydot(temp_vec, line_vec);

            /// foot position of the joint position on the path line
            raisim::Vec<3> foot_pos = { m_PathPosGlobal[m_nTargetPath[iter][1] - 1][0] + child_to_foot_vec[0],
                                       m_PathPosGlobal[m_nTargetPath[iter][1] - 1][1] + child_to_foot_vec[1],
                                       m_PathPosGlobal[m_nTargetPath[iter][1] - 1][2] + child_to_foot_vec[2] };

            /// calculation of vector from joint position to path_line
            raisim::Vec<3> joint_to_foot = { foot_pos[0] - global_joint_pos[0],
                                            foot_pos[1] - global_joint_pos[1],
                                            foot_pos[2] - global_joint_pos[2] };

            /// joint axis in global coordinate
            raisim::Vec<3> JointAxis;
            raisim::Mat<3, 3> RotMat{};
            model->getJoint(m_strTargetJoint[iter]).getPose(JointAxis, RotMat);

            /// it should be changed to joint axis (2020.08.03)
            if (m_strTargetJoint[iter] == "subtalar_r")
                JointAxis = { 0.78717961000000003, 0.60474746000000001, -0.12094949000000001 };
            else if (m_strTargetJoint[iter] == "mtp_r")
                JointAxis = { -0.58095439999999998, 0, 0.81393610999999999 };
            else if (m_strTargetJoint[iter] == "subtalar_l")
                JointAxis = { -0.78717961000000003, -0.60474746000000001, -0.12094949000000001 };
            else if (m_strTargetJoint[iter] == "mtp_l")
                JointAxis = { 0.58095439999999998, 0, 0.81393610999999999 };
            else if (m_strTargetJoint[iter] == "hip_r")
                JointAxis = { -sin(-0.25), 0, cos(-0.25) };
            else if (m_strTargetJoint[iter] == "hip_l")
                JointAxis = { -sin(0.25), 0, cos(0.25) };
            else
                JointAxis = { 0., 0., 1. };
            /// it should be changed to joint axis (2020.08.03)

            /// axis of the joint in global frame
            raisim::Vec<3> global_joint = {
                    RotMat[0] * JointAxis[0] + RotMat[3] * JointAxis[1] + RotMat[6] * JointAxis[2],
                    RotMat[1] * JointAxis[0] + RotMat[4] * JointAxis[1] + RotMat[7] * JointAxis[2],
                    RotMat[2] * JointAxis[0] + RotMat[5] * JointAxis[1] + RotMat[8] * JointAxis[2] };

            raisim::Vec<3> third_vec = mycross(global_joint, line_vec);
            raisim::Vec<3> third_vec_normalized = third_vec / third_vec.norm();
            raisim::Vec<3> joint_to_foot_temp = third_vec * mydot(third_vec_normalized, joint_to_foot);

            /// cross product of foot vector and joint axis
            raisim::Vec<3> cross = mycross(joint_to_foot_temp, line_vec);

            if (mydot(cross, global_joint) > 0)
                m_moment_arm[iter] = joint_to_foot_temp.norm();
            else
                m_moment_arm[iter] = -joint_to_foot_temp.norm();

            /// spherical joint
            if (m_strTargetJoint[iter] == "hip_r" || m_strTargetJoint[iter] == "hip_l" || m_strTargetJoint[iter] == "back") {
                for (int ispherical = 0; ispherical < 2; ispherical++) {
                    if (ispherical == 0)
                        if (m_strTargetJoint[iter] == "hip_r")
                            JointAxis = { cos(-0.25), 0, sin(-0.25) };
                        else if (m_strTargetJoint[iter] == "hip_l")
                            JointAxis = { cos(0.25), 0, sin(0.25) };
                        else
                            JointAxis = { 1., 0., 0. };
                    else
                        JointAxis = { 0, 1, 0 };
                    /// axis of the joint in global frame
                    raisim::Vec<3> global_joint_2 = {
                            RotMat[0] * JointAxis[0] + RotMat[3] * JointAxis[1] + RotMat[6] * JointAxis[2],
                            RotMat[1] * JointAxis[0] + RotMat[4] * JointAxis[1] + RotMat[7] * JointAxis[2],
                            RotMat[2] * JointAxis[0] + RotMat[5] * JointAxis[1] + RotMat[8] * JointAxis[2] };
                    /// moment arm
                    third_vec = mycross(global_joint_2, line_vec);
                    third_vec_normalized = third_vec / third_vec.norm();
                    joint_to_foot_temp = third_vec * mydot(third_vec_normalized, joint_to_foot);
                    cross = mycross(joint_to_foot_temp, line_vec);

                    if (mydot(cross, global_joint_2) > 0)
                        m_moment_arm_spherical[iter][ispherical] = joint_to_foot_temp.norm();
                    else
                        m_moment_arm_spherical[iter][ispherical] = -joint_to_foot_temp.norm();
                }
            }
        }

        /// nan check for conditional path point
        if (isnan(m_moment_arm[iter]))
            m_moment_arm[iter] = 0.;
        if (isnan(m_moment_arm_spherical[iter][0]))
            m_moment_arm_spherical[iter][0] = 0.;
        if (isnan(m_moment_arm_spherical[iter][1]))
            m_moment_arm_spherical[iter][1] = 0.;
    }
}


void MuscleModel::ApplyForce(std::vector<double> &ForceTable) {
    for (int iter = 0; iter < m_nTargetJoint; iter++) {
        size_t t = model->getJoint(m_strTargetJoint[iter]).getIdxInGeneralizedCoordinate();

        if (m_strTargetJoint[iter] == "hip_r" || m_strTargetJoint[iter] == "hip_l" || m_strTargetJoint[iter] == "back") {
            ForceTable[t] += m_TendonForce * m_moment_arm_spherical[iter][0];
            ForceTable[t + 1] += m_TendonForce * m_moment_arm_spherical[iter][1];
            ForceTable[t + 2] += m_TendonForce * m_moment_arm[iter]; assert(t+2 < 31);
        }

        else {
            ForceTable[t] += m_TendonForce * m_moment_arm[iter];
            assert(t >= 0);
            assert(t < 31);
        }
    }
}

void MuscleModel::SetScaleLength(double scale) {
    m_constTendonSlackLength *= scale;
    m_const_OptFiberLength *= scale;
}

void MuscleModel::SetScaleMuscle(
        const std::vector<std::string> &scale_link,
        const std::vector<double> &scale_value,
        const std::vector<raisim::Vec<3>> &trs,
        const std::vector<raisim::Mat<3, 3>> &rot,
        double l0) {

    /// pre-scaled length
    double length_temp = l0;

    /// calculate scaled path positions
    int idx = 0;
    for (int iter = 0; iter < m_nPathPoint; iter++) {
        /// common and conditional path points
        for (int j = 0; j < 20; j++) {
            if (m_PathLink[iter] == scale_link[j]) {
                idx = j;
                break;  // by SKOO
            }
        }
        m_PathPosLocal[iter][0] *= scale_value[idx];
        m_PathPosLocal[iter][1] *= scale_value[idx];
        m_PathPosLocal[iter][2] *= scale_value[idx];

        /// moving path points
        for (int j = 0; j < vy.size(); j++)
            vy[j] *= scale_value[idx];
        for (int j = 0; j < vy2.size(); j++)
            vy2[j] *= scale_value[idx];
    }

    std::vector<double> b_temp, c_temp, d_temp;
    b_temp.clear();c_temp.clear();d_temp.clear();
    calcCoefficients(vx, vy, b_temp, c_temp, d_temp);
    x_b.clear();x_c.clear();x_d.clear();
    for (int j = 0; j < vx.size(); j++) {
        x_b.push_back(b_temp.at(j));
        x_c.push_back(c_temp.at(j));
        x_d.push_back(d_temp.at(j));
    }
    b_temp.clear();c_temp.clear();d_temp.clear();
    calcCoefficients(vx2, vy2, b_temp, c_temp, d_temp);
    y_b.clear();y_c.clear();y_d.clear();
    for (int j = 0; j < vx.size(); j++) {
        y_b.push_back(b_temp.at(j));
        y_c.push_back(c_temp.at(j));
        y_d.push_back(d_temp.at(j));
    }

    /// calculate scaled length
    UpdateGlobalPos(trs, rot);
    CalcTotalLength();
    double length_ratio = m_muscleLength / length_temp;
    SetScaleLength(length_ratio);
}

double MuscleModel::CalcPennationAngle(double dTendonLength) {
    double dPennationAngle = -1.0;

    if (m_const_OptPennationAngle > 0.0) {
        double cos_fiber_length = m_muscleLength - dTendonLength;
        if (m_const_muscle_height/cos_fiber_length > 9.0)
            dPennationAngle = atan(9.0);
        else {
            double dFiberLength = sqrt(cos_fiber_length*cos_fiber_length + m_const_muscle_height * m_const_muscle_height);
            dPennationAngle = asin(m_const_muscle_height / dFiberLength);
        }
    }
    else
        dPennationAngle = 0.0;

    return dPennationAngle;
}

/// calculate initial length of a muscle
void MuscleModel::initialize(const std::vector<raisim::Vec<3>> &trs, const std::vector<raisim::Mat<3, 3>> &rot) {
    UpdateGlobalPos(trs, rot);
    CalcTotalLength();

    InitLength = m_muscleLength;

    if (m_const_OptPennationAngle > 0.0) {  // SKOO 20210619 It is always 0.0. It needs check.
        m_const_muscle_height = m_const_OptFiberLength * sin(m_const_OptPennationAngle);
        m_const_minimum_fiber_length = m_const_muscle_height / sin(atan(9.0));
    }
    else { // in case that the pennation angle is zero
        m_const_muscle_height = 0.0;
        m_const_minimum_fiber_length = 0.01*m_const_OptFiberLength;
    }

    m_currentFiberLength = m_const_OptFiberLength;
    m_currentTendonLength = m_constTendonSlackLength;
}

/// activation dynamics to calculate activation from excitation (Thelen 2003)
void MuscleModel::CalcActivation() {
    double da_over_dt = 0.0;

    if (m_muscleExcitation < 0.0)
        m_muscleExcitation = 0;
    else if (m_muscleExcitation > 1.0)
        m_muscleExcitation = 1.0;

    if (m_muscleActivation == -0.01) // set activation at start frame
        m_muscleActivation = m_muscleExcitation;
    else { // calculate activation during episode
        if (m_muscleExcitation > m_muscleActivation)
            da_over_dt = (m_muscleExcitation - m_muscleActivation) / (0.01 * (0.5 + 1.5 * m_muscleActivation));
        else
            da_over_dt = (m_muscleExcitation - m_muscleActivation) / (0.04 / (0.5 + 1.5 * m_muscleActivation));

        m_muscleActivation += da_over_dt * m_const_control_dt;
    }

    // muscle activation clipping
    if (m_muscleActivation < 0.0)
        m_muscleActivation = 0.0;
    else if (m_muscleActivation > 1.0)
        m_muscleActivation = 1.0;
}

void MuscleModel::calcCoefficients(const std::vector<double> &x, const std::vector<double> &y, std::vector<double> &b, std::vector<double> &c, std::vector<double> &d) {
    int n = x.size();
    int nm1, nm2, i, j;
    double t;

    if (n < 2)
        return;  // SKOO what does it mean?
    assert(n > 3);

    std::vector<double> _b, _c, _d, _x, _y;
    for (int iter = 0; iter < n; iter++) {
        _b.push_back(0.);
        _c.push_back(0.);
        _d.push_back(0.);
    }

    _x = x;
    _y = y;

    nm1 = n - 1;
    nm2 = n - 2;

    _d[0] = _x[1] - _x[0];
    _c[1] = (_y[1] - _y[0]) / _d[0];
    for (i = 1; i < nm1; i++)
    {
        _d[i] = _x[i + 1] - _x[i];
        _b[i] = 2.0*(_d[i - 1] + _d[i]);
        _c[i + 1] = (_y[i + 1] - _y[i]) / _d[i];
        _c[i] = _c[i + 1] - _c[i];
    }

    /* End conditions. Third derivatives at x[0] and x[n-1]
     * are obtained from divided differences.
     */

    _b[0] = -_d[0];
    _b[nm1] = -_d[nm2];
    _c[0] = 0.0;
    _c[nm1] = 0.0;

    if (n > 3)
    {
        double d1, d2, d3, d20, d30, d31;

        d31 = _x[3] - _x[1];
        d20 = _x[2] - _x[0];
        d1 = _x[nm1] - _x[n - 3];
        d2 = _x[nm2] - _x[n - 4];
        d30 = _x[3] - _x[0];
        d3 = _x[nm1] - _x[n - 4];
        _c[0] = _c[2] / d31 - _c[1] / d20;
        _c[nm1] = _c[nm2] / d1 - _c[n - 3] / d2;
        _c[0] = _c[0] * _d[0] * _d[0] / d30;
        _c[nm1] = -_c[nm1] * _d[nm2] * _d[nm2] / d3;
    }

    /* Forward elimination */

    for (i = 1; i < n; i++)
    {
        t = _d[i - 1] / _b[i - 1];
        _b[i] -= t * _d[i - 1];
        _c[i] -= t * _c[i - 1];
    }

    /* Back substitution */

    _c[nm1] /= _b[nm1];
    for (j = 0; j < nm1; j++)
    {
        i = nm2 - j;
        _c[i] = (_c[i] - _d[i] * _c[i + 1]) / _b[i];
    }

    /* compute polynomial coefficients */

    _b[nm1] = (_y[nm1] - _y[nm2]) / _d[nm2] +
        _d[nm2] * (_c[nm2] + 2.0*_c[nm1]);
    for (i = 0; i < nm1; i++)
    {
        _b[i] = (_y[i + 1] - _y[i]) / _d[i] - _d[i] * (_c[i + 1] + 2.0*_c[i]);
        _d[i] = (_c[i + 1] - _c[i]) / _d[i];
        _c[i] *= 3.0;
    }
    _c[nm1] *= 3.0;
    _d[nm1] = _d[nm2];

    for (int iter = 0; iter < n; iter++) {
        b.push_back(_b[iter]);
        c.push_back(_c[iter]);
        d.push_back(_d[iter]);
    }
    //std::cout << b.at(0) << std::endl;
}

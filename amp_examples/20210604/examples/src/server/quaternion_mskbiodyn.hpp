#ifndef _QUATERNION_MSKBIODYN_H_
#define _QUATERNION_MSKBIODYN_H_

/*
Copyright (c) 2021 Seungbum Koo, PhD, KAIST
MIT License

Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

Except as contained in this notice, the name(s) of the above
copyright holders shall not be used in advertising or otherwise
to promote the sale, use or other dealings in this Software
without prior written authorization.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
=====================================================

Quaternion C++ class.
Ver 1.0, 2020.07.14

Seungbum Koo, PhD
Musculoskeletal BioDynamics Lab
Korea Advanced Institute of Science and Technology
skoo@kaist.ac.kr

q = w + x * i + y * j + z * k

If using quaternions to represent rotations, q must be a unit quaternion.
Then the following relates q to the corresponding rotation:
    s = cos(angle/2)
    (x,y,z) = sin(angle/2) * axis_of_rotation
    (axis_of_rotation is unit length)
*/

#include <cmath>

class Quaternion {
public:
    Quaternion(double w, double x, double y, double z) {
        m_w = w; m_x= x; m_y = y; m_z = z;
    }

    Quaternion() {
        m_w = 1, m_x = 0, m_y = 0, m_z = 0;
    }

    double getW() { return m_w; }
    double getX() { return m_x; }
    double getY() { return m_y; }
    double getZ() { return m_z; }
    double norm() {
        return sqrt(m_w*m_w + m_x*m_x + m_y*m_y + m_z*m_z); }
    void set(double w, double x, double y, double z) {
        m_w = w; m_x= x; m_y = y; m_z = z;
    }

    Quaternion &operator= (Quaternion rhs);
    bool operator== (Quaternion rhs);
    bool operator!= (Quaternion rhs);

    Quaternion operator+ (Quaternion q2) const;
    Quaternion operator- (Quaternion q2) const;
    Quaternion operator* (Quaternion q2) const;
    Quaternion operator* (double a) const;
    Quaternion operator/ (Quaternion q2) const;
    Quaternion operator/ (double a) const;

    Quaternion conj();
    Quaternion inv();

    Quaternion log();
    Quaternion get_normalized();

    void Normalize();

    static double get_dot_two_quaternions(Quaternion &q1, Quaternion &q2);
    static double get_angle_two_quaternions(Quaternion &q1, Quaternion &q2);
    static void get_log_quat_diff(double *v1, const double *q1v, const double *q2v);
    static void rotvec2quat(double *q1, const double *v1);
    static void quat2rotvec(double *v1, const double *q1);
    static void slerp(Quaternion &qout, Quaternion &q1, Quaternion &q2, double lambda);
    // static void get_box_minus_two_quaternions(double *rotvel, Quaternion &q1, Quaternion &q2);

private:
    double m_w = 1.0, m_x = 0.0, m_y = 0.0, m_z = 0.0;
};


void Quaternion::rotvec2quat(double *q1, const double *v1) {

    double qrot; // rotation angle in radian
    double arot[3]; // rotation axis

    qrot = sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
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


Quaternion Quaternion::inv() {
    Quaternion q1 = *this;
    Quaternion q2;
    q2 = q1.conj()/(q1.norm()*q1.norm());
    return q2;
}


Quaternion Quaternion::log() {
    double theta = acos(m_w)*2.0;
    Quaternion q(0, theta*m_x, theta*m_y, theta*m_z);
    return q;
}


void Quaternion::quat2rotvec(double *v1, const double *q1) {
    double qrot; // rotation angle in radian
    double arot[3]; // rotation axis

    qrot = acos(q1[0]) * 2.0;

    if (qrot < 0.001) {
        v1[0] = 0.0;
        v1[1] = 0.0;
        v1[2] = 0.0;
    } else {
        arot[0] = q1[1] / sin(qrot / 2.0);
        arot[1] = q1[2] / sin(qrot / 2.0);
        arot[2] = q1[3] / sin(qrot / 2.0);

        v1[0] = arot[0] * qrot;
        v1[1] = arot[1] * qrot;
        v1[2] = arot[2] * qrot;
    }
}


void Quaternion::slerp(Quaternion &qout, Quaternion &q1, Quaternion &q2, double lambda) {

    double w1, x1, y1, z1, norm1;
    double w2, x2, y2, z2, norm2;
    double w3, x3, y3, z3, norm3;

    // SKOO 20200711 https://en.wikipedia.org/wiki/Slerp

    norm1 = q1.norm();
    w1 = q1.getW()/norm1; x1 = q1.getX()/norm1; y1 = q1.getY()/norm1; z1 = q1.getZ()/norm1;

    norm2 = q2.norm();
    w2 = q2.getW()/norm2; x2 = q2.getX()/norm2; y2 = q2.getY()/norm2; z2 = q2.getZ()/norm2;

    // Reverse the sign of q2 if dot(q1, q2) < 0.
    double dotq1q2 = w1*w2 + x1*x2 + y1*y2 + z1*z2;
    if (dotq1q2 < 0) {
        w2 = -w2;
        x2 = -x2;
        y2 = -y2;
        z2 = -z2;
        dotq1q2 = -1*dotq1q2;
    }

    const double DOT_THRESHOLD = 0.9995;
    if (dotq1q2 > DOT_THRESHOLD) {
        // If the inputs are too close for comfort, linearly interpolate
        // and normalize the result.
        w3 = w1 + lambda*(w2 - w1);
        x3 = x1 + lambda*(x2 - x1);
        y3 = y1 + lambda*(y2 - y1);
        z3 = z1 + lambda*(z2 - z1);
        norm3 = sqrt(w3*w3 + x3*x3 + y3*y3 + z3*z3);
        w3 /= norm3; x3 /= norm3; y3 /= norm3; z3 /= norm3;
        qout.set(w3, x3, y3, z3);
        return;
    }

    // Since dotq1q2 is in range [0, DOT_THRESHOLD], acos is safe
    double theta_0 = acos(dotq1q2);     // theta_0 = angle between input vectors
    double theta = theta_0*lambda;      // theta = angle between v0 and result
    double sin_theta = sin(theta);      // compute this value only once
    double sin_theta_0 = sin(theta_0);  // compute this value only once

    double s0 = cos(theta) - dotq1q2 * sin_theta / sin_theta_0;  // == sin(theta_0 - theta) / sin(theta_0)
    double s1 = sin_theta / sin_theta_0;

    w3 = s0*w1 + s1*w2;
    x3 = s0*x1 + s1*x2;
    y3 = s0*y1 + s1*y2;
    z3 = s0*z1 + s1*z2;
    qout.set(w3, x3, y3, z3);
}


Quaternion &Quaternion::operator= (Quaternion rhs) {

    m_w = rhs.getW(); m_x = rhs.getX(); m_y = rhs.getY(); m_z = rhs.getZ();
    return *this;
}


bool Quaternion::operator== (Quaternion rhs) {

    return ((m_w == rhs.getW()) && (m_x == rhs.getX()) && (m_y == rhs.getY()) && (m_z == rhs.getZ()));
}


bool Quaternion::operator!= (Quaternion rhs) {

    return ((m_w != rhs.getW()) || (m_x != rhs.getX()) || (m_y != rhs.getY()) || (m_z != rhs.getZ()));
}


Quaternion Quaternion::operator+ (Quaternion q2) const {

    Quaternion q(m_w + q2.getW(), m_x + q2.getX(), m_y + q2.getY(), m_z + q2.getZ());
    return q;
}


Quaternion Quaternion::operator- (Quaternion q2) const {

    Quaternion q(m_w - q2.getW(), m_x - q2.getX(), m_y - q2.getY(), m_z - q2.getZ());
    return q;
}


Quaternion Quaternion::operator* (Quaternion q2) const {

    Quaternion q(
        m_w * q2.getW() - m_x * q2.getX() - m_y * q2.getY() - m_z * q2.getZ(),
        m_w * q2.getX() + m_x * q2.getW() + m_y * q2.getZ() - m_z * q2.getY(),
        m_w * q2.getY() - m_x * q2.getZ() + m_y * q2.getW() + m_z * q2.getX(),
        m_w * q2.getZ() + m_x * q2.getY() - m_y * q2.getX() + m_z * q2.getW());

    return q;
}


Quaternion Quaternion::operator* (double a) const {

    Quaternion q(a*m_w, a*m_x, a*m_y, a*m_z);
    return q;
}


Quaternion Quaternion::operator/ (Quaternion q2) const {

    Quaternion inv_q2 = q2.conj()/(q2.norm()*q2.norm());
    Quaternion q = *this * inv_q2;
    return q;
}


Quaternion Quaternion::operator/ (double a) const {

    Quaternion q(m_w/a, m_x/a, m_y/a, m_z/a);
    return q;
}


Quaternion Quaternion::conj() {

    Quaternion q(m_w, -m_x, -m_y, -m_z);
    return q;
}


Quaternion Quaternion::get_normalized() {

    Quaternion q = *this;
    return q/q.norm();
}


void Quaternion::Normalize() {

    double qnorm = norm();
    m_w /= qnorm; m_x /= qnorm; m_y /= qnorm; m_z /= qnorm;
}


double Quaternion::get_dot_two_quaternions(Quaternion &q1, Quaternion &q2) {

    return q1.getW()*q2.getW() + q1.getX()*q2.getX() + q1.getY()*q2.getY() + q1.getZ()*q2.getZ();
}


double Quaternion::get_angle_two_quaternions(Quaternion &q1, Quaternion &q2) {

    Quaternion q3 = q2*q1.inv();
    Quaternion q4 = q3.log();

    return q4.norm();
}

void Quaternion::get_log_quat_diff(double *v1, const double *q1v, const double *q2v) {
    Quaternion q1(q1v[0], q1v[1], q1v[2], q1v[3]);
    Quaternion q2(q2v[0], q2v[1], q2v[2], q2v[3]);
    Quaternion q3 = q1.inv()*q2; // or q2*q1.inv()
    Quaternion q4 = q3.log();
    v1[0] = q4.getX();
    v1[1] = q4.getY();
    v1[2] = q4.getZ();
}


/*
// box minus operation in KINDR library
void Quaternion::get_box_minus_two_quaternions(double *rotvel, Quaternion &q1, Quaternion &q2)
{
    double w1, x1, y1, z1, norm1;
    double w2, x2, y2, z2, norm2;
    double w3, x3, y3, z3;

    // SKOO 20200711 Box Minus Operation in KINDR library
    // rotvel = logmap(q1 x inv(q2))

    norm1 = q1.norm();
    w1 = q1.getW() / norm1; x1 = q1.getX() / norm1; y1 = q1.getY() / norm1; z1 = q1.getZ() / norm1;

    norm2 = q2.norm();
    w2 = q2.getW() / norm2; x2 = q2.getX() / norm2; y2 = q2.getY() / norm2; z2 = q2.getZ() / norm2;

    double dotq1q2 = w1*w2 + x1*x2 + y1*y2 + z1*z2; 
    // get_dot_two_quaternions(q1.get_normalized(), q2.get_normalized());

    const double DOT_THRESHOLD = 0.9995;
    if (dotq1q2 > DOT_THRESHOLD) {
        // If the inputs are too close for comfort,
        // the rotational velocity is close to zero.
        rotvel[0] = 0;
        rotvel[1] = 0;
        rotvel[2] = 0;
        return;
    }

    // q3 = q1 * inv(q2)
    w3 =  w1*w2 + x1*x2 + y1*y2 + z1*z2;
    x3 = -w1*x2 + x1*w2 - y1*z2 + z1*y2;
    y3 = -w1*y2 + x1*z2 + y1*w2 - z1*x2;
    z3 = -w1*z2 - x1*y2 + y1*x2 + z1*w2;

    double qvnorm = sqrt(x3*x3 + y3*y3 + z3*z3);
    double qwacos = acos(w3 - 0.000001);
    rotvel[0] = 2*qwacos/qvnorm*x3;
    rotvel[1] = 2*qwacos/qvnorm*y3;
    rotvel[2] = 2*qwacos/qvnorm*z3;
}
*/

/*
// Transforms the quaternion to the corresponding rotation matrix.
// Quaternion is assumed to be a unit quaternion.
// R is a 3x3 orthogonal matrix and will be returned in row-major order.
template <typename real>
inline void Quaternion<real>::Quaternion2Matrix(real * R)
{
    R[0] = 1 - 2 * y*y - 2 * z*z; R[1] = 2 * x*y - 2 * s*z;     R[2] = 2 * x*z + 2 * s*y;
    R[3] = 2 * x*y + 2 * s*z;     R[4] = 1 - 2 * x*x - 2 * z*z; R[5] = 2 * y*z - 2 * s*x;
    R[6] = 2 * x*z - 2 * s*y;     R[7] = 2 * y*z + 2 * s*x;     R[8] = 1 - 2 * x*x - 2 * y*y;
}


// Returns (x,y,z) = sin(theta/2) * axis, where
//   theta is the angle of rotation, theta\in\{-pi,pi\}, and
//   axis is the unit axis of rotation.
template <typename real>
inline void Quaternion<real>::GetSinExponential(real * sex, real * sey, real * sez)
{
    if (s<0)
    {
        *sex = -x;
        *sey = -y;
        *sez = -z;
    }
    else
    {
        *sex = x;
        *sey = y;
        *sez = z;
    }
}

template <typename real>
inline void Quaternion<real>::GetRotation(real * angle, real unitAxis[3])
{
    if ((s >= ((real)1)) || (s <= (real)(-1)))
    {
        // identity; this check is necessary to avoid problems with acos if s is 1 + eps
        *angle = 0;
        unitAxis[0] = 1;
        unitAxis[0] = 0;
        unitAxis[0] = 0;
        return;
    }

    *angle = 2.0 * acos(s);
    real sin2 = x*x + y*y + z*z; //sin^2(*angle / 2.0)

    if (sin2 == 0)
    {
        // identity rotation; angle is zero, any axis is equally good
        unitAxis[0] = 1;
        unitAxis[0] = 0;
        unitAxis[0] = 0;
    }
    else
    {
        real inv = 1.0 / sqrt(sin2); // note: *angle / 2.0 is on [0,pi], so sin(*angle / 2.0) >= 0, and therefore the sign of sqrt can be safely taken positive
        unitAxis[0] = x * inv;
        unitAxis[1] = y * inv;
        unitAxis[2] = z * inv;
    }
}

template <typename real>
inline void Quaternion<real>::MoveToRightHalfSphere()
{
    if (s<0)
    {
    s *= -1;
    x *= -1;
    y *= -1;
    z *= -1;
    }
}

template <typename real>
inline void Quaternion<real>::Print()
{
    printf("%f + %fi + %fj + %fk\n", s, x, y, z);
}
*/


/*
std::ostream& operator<<( std::ostream& os, quater const& a )
{
os << "( " << a.p[0] << " , " << a.p[1] << " , " << a.p[2] << " , " << a.p[3] << " )";
return os;
}

std::istream& operator>>( std::istream& is, quater& a )
{
static char	buf[256];
//is >> "(" >> a.p[0] >> "," >> a.p[1] >> "," >> a.p[2] >> "," >> a.p[3] >> ")";
is >> buf >> a.p[0] >> buf >> a.p[1] >> buf >> a.p[2] >> buf >> a.p[3] >> buf;
return is;
}

quater exp(vector const& w)
{
m_real theta = sqrt(w % w);
m_real sc;

if(theta < eps) sc = 1;
else sc = sin(theta) / theta;

vector v = sc * w;
return quater(cos(theta), v.x(), v.y(), v.z());
}

quater pow(vector const& w, m_real a)
{
return exp(a * w);
}

vector ln(quater const& q)
{
m_real sc = sqrt(q.p[1] * q.p[1] + q.p[2] * q.p[2] + q.p[3] * q.p[3]);
m_real theta = atan2(sc, q.p[0]);
if(sc > eps)
sc = theta / sc;
else  sc = 1.0 ;
return vector(sc * q.p[1], sc * q.p[2], sc * q.p[3]);
}

position rotate(quater const& a, position const& v)
{
quater c = a * quater(0, v.x(), v.y(), v.z()) * inverse(a);
return position(c.x(), c.y(), c.z());
}

vector rotate(quater const& a, vector const& v)
{
quater c = a * quater(0, v.x(), v.y(), v.z()) * inverse(a);
return vector(c.x(), c.y(), c.z());
}

unit_vector rotate(quater const& a, unit_vector const& v)
{
quater c = a * quater(0, v.x(), v.y(), v.z()) * inverse(a);
return unit_vector(c.x(), c.y(), c.z());
}

quater
slerp( quater const& a, quater const& b, m_real t )
{
m_real c = a % b;

if ( 1.0+c > EPS_jhm )
{
if ( 1.0-c > EPS_jhm )
{
m_real theta = acos( c );
m_real sinom = sin( theta );
return ( a*sin((1-t)*theta) + b*sin(t*theta) ) / sinom;
}
else
return (a*(1-t) + b*t).normalize();
}
else	return a*sin((0.5-t)*M_PI) + b*sin(t*M_PI);
}

quater
interpolate( m_real t, quater const& a, quater const& b )
{
return slerp( a, b, t );
}

m_real
distance( quater const& a, quater const& b )
{
return MIN( ln( a.inverse()* b).length(),
ln( a.inverse()*-b).length() );
}

vector
difference( quater const& a, quater const& b )
{
return ln( b.inverse() * a );
}

}


*/


#endif
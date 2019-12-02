#ifndef UTILS_H
#define UTILS_H

#include <QObject>
#include <QDebug>
#include <QDataStream>

#include <Eigen/Core>

class Utils
{
public:
    Utils();
};

template<typename _Scalar, int _Rows, int _Cols>
QDebug &qDebugMatrix(QDebug &out, Eigen::Matrix<_Scalar, _Rows, _Cols> m);

template<typename _Scalar, int _Rows, int _Cols>
QDataStream &streamInMatrix(QDataStream &in, Eigen::Matrix<_Scalar, _Rows, _Cols> &m);

QDebug &operator<<(QDebug out, Eigen::Matrix4f m);

QDataStream &operator>>(QDataStream &in, Eigen::Matrix4f &m);

#endif // UTILS_H

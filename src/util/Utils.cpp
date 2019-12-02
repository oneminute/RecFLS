#include "Utils.h"

Utils::Utils()
{

}

template<typename _Scalar, int _Rows, int _Cols>
QDebug &qDebugMatrix(QDebug &out, Eigen::Matrix<_Scalar, _Rows, _Cols> m)
{
    for (int i = 0; i < m.rows(); i++)
    {
        for (int j = 0; j < m.cols(); j++)
        {
            out << fixed << qSetRealNumberPrecision(2) << qSetFieldWidth(6) << m.row(i)[j];
        }
        out << endl;
    }
    return out;
}

template<typename _Scalar, int _Rows, int _Cols>
QDataStream &streamInMatrix(QDataStream &in, Eigen::Matrix<_Scalar, _Rows, _Cols> &m)
{
    for (int i = 0; i < m.size(); i++)
    {
        float v;
        in >> v;
        m.data()[i] = v;
    }
    return in;
}

QDebug &operator<<(QDebug out, Eigen::Matrix4f m)
{
    return qDebugMatrix(out, m);
}

QDataStream &operator>>(QDataStream &in, Eigen::Matrix4f &m)
{
    return streamInMatrix(in, m);
}

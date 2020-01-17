#include "Utils.h"
#include "common/Frame.h"

#include <pcl/common/eigen.h>
#include <pcl/common/common.h>
#include <QImage>

Utils::Utils()
{

}

void Utils::registerTypes()
{
    qRegisterMetaType<Frame>("Frame&");
}

template<typename _Scalar, int _Rows, int _Cols>
QDebug qDebugMatrix(QDebug &out, const Eigen::Matrix<_Scalar, _Rows, _Cols> &m)
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

QDebug operator<<(QDebug out, const Eigen::Matrix4f &m)
{
    return qDebugMatrix(out, m);
}

QDataStream &operator>>(QDataStream &in, Eigen::Matrix4f &m)
{
    return streamInMatrix(in, m);
}

QDebug operator<<(QDebug out, const Eigen::Vector3f &v)
{
    return qDebugMatrix(out, v);
}

QImage cvMat2QImage(const cv::Mat &image, bool isBgr, uCvQtDepthColorMap colorMap)
{
    QImage qtemp;
    if(!image.empty() && image.depth() == CV_8U)
    {
        if(image.channels()==3)
        {
            const unsigned char * data = image.data;
            if(image.channels() == 3)
            {
                qtemp = QImage(image.cols, image.rows, QImage::Format_RGB32);
                for(int y = 0; y < image.rows; ++y, data += image.cols*image.elemSize())
                {
                    for(int x = 0; x < image.cols; ++x)
                    {
                        QRgb * p = ((QRgb*)qtemp.scanLine (y)) + x;
                        if(isBgr)
                        {
                            *p = qRgb(data[x * image.channels()+2], data[x * image.channels()+1], data[x * image.channels()]);
                        }
                        else
                        {
                            *p = qRgb(data[x * image.channels()], data[x * image.channels()+1], data[x * image.channels()+2]);
                        }
                    }
                }
            }
        }
        else if(image.channels() == 1)
        {
            // mono grayscale
            qtemp = QImage(image.data, image.cols, image.rows, image.cols, QImage::Format_Indexed8).copy();
            QVector<QRgb> my_table;
            for(int i = 0; i < 256; i++)
                my_table.push_back(qRgb(i,i,i));
            qtemp.setColorTable(my_table);
        }
        else
        {
            printf("Wrong image format, must have 1 or 3 channels\n");
        }
    }
    else if(image.depth() == CV_32F && image.channels()==1)
    {
        // Assume depth image (float in meters)
        const float * data = (const float *)image.data;
        float min=data[0], max=data[0];
        for(unsigned int i=1; i<image.total(); ++i)
        {
            if(qIsFinite(data[i]) && data[i] > 0)
            {
                if(!qIsFinite(min) || (data[i] > 0 && data[i]<min))
                {
                    min = data[i];
                }
                if(!qIsFinite(max) || (data[i] > 0 && data[i]>max))
                {
                    max = data[i];
                }
            }
        }

        qtemp = QImage(image.cols, image.rows, QImage::Format_Indexed8);
        for(int y = 0; y < image.rows; ++y, data += image.cols)
        {
            for(int x = 0; x < image.cols; ++x)
            {
                uchar * p = qtemp.scanLine (y) + x;
                if(data[x] < min || data[x] > max || !qIsFinite(data[x]) || max == min)
                {
                    *p = 0;
                }
                else
                {
                    *p = uchar(255.0f - ((data[x]-min)*255.0f)/(max-min));
                    if(*p == 255)
                    {
                        *p = 0;
                    }
                }
                if(*p!=0 && (colorMap == uCvQtDepthBlackToWhite || colorMap == uCvQtDepthRedToBlue))
                {
                    *p = 255-*p;
                }
            }
        }

        QVector<QRgb> my_table;
        my_table.reserve(256);
        if(colorMap == uCvQtDepthRedToBlue || colorMap == uCvQtDepthBlueToRed)
        {
            my_table.push_back(qRgb(0,0,0));
            for(int i = 1; i < 256; i++)
                my_table.push_back(QColor::fromHsv(i, 255, 255, 255).rgb());
        }
        else
        {
            for(int i = 0; i < 256; i++)
                my_table.push_back(qRgb(i,i,i));
        }
        qtemp.setColorTable(my_table);
    }
    else if(image.depth() == CV_16U && image.channels()==1)
    {
        // Assume depth image (unsigned short in mm)
        const unsigned short * data = (const unsigned short *)image.data;
        unsigned short min=data[0], max=data[0];
        for(unsigned int i=1; i<image.total(); ++i)
        {
            if(qIsFinite(static_cast<double>(data[i])) && data[i] > 0)
            {
                if(!qIsFinite(static_cast<double>(min)) || (data[i] > 0 && data[i]<min))
                {
                    min = data[i];
                }
                if(!qIsFinite(static_cast<double>(max)) || (data[i] > 0 && data[i]>max))
                {
                    max = data[i];
                }
            }
        }

        qtemp = QImage(image.cols, image.rows, QImage::Format_Indexed8);
        for(int y = 0; y < image.rows; ++y, data += image.cols)
        {
            for(int x = 0; x < image.cols; ++x)
            {
                uchar * p = qtemp.scanLine (y) + x;
                if(data[x] < min || data[x] > max || !qIsFinite(static_cast<double>(data[x])) || max == min)
                {
                    *p = 0;
                }
                else
                {
                    *p = uchar(255.0f - (float(data[x]-min)/float(max-min))*255.0f);
                    if(*p == 255)
                    {
                        *p = 0;
                    }
                }
                if(*p!=0 && (colorMap == uCvQtDepthBlackToWhite || colorMap == uCvQtDepthRedToBlue))
                {
                    *p = 255-*p;
                }
            }
        }

        QVector<QRgb> my_table;
        my_table.reserve(256);
        if(colorMap == uCvQtDepthRedToBlue || colorMap == uCvQtDepthBlueToRed)
        {
            my_table.push_back(qRgb(0,0,0));
            for(int i = 1; i < 256; i++)
                my_table.push_back(QColor::fromHsv(i, 255, 255, 255).rgb());
        }
        else
        {
            for(int i = 0; i < 256; i++)
                my_table.push_back(qRgb(i,i,i));
        }
        qtemp.setColorTable(my_table);
    }
    else if(!image.empty() && image.depth() != CV_8U)
    {
        printf("Wrong image format, must be 8_bits/3channels or (depth) 32bitsFloat/1channel, 16bits/1channel\n");
    }
    return qtemp;
}

Eigen::Matrix4f matrix4fFrom(float x, float y, float z, float roll, float pitch, float yaw)
{
    Eigen::Affine3f t = pcl::getTransformation(x, y, z, roll, pitch, yaw);
    return t.matrix();
}

Eigen::Matrix4f matrix4fFrom(float x, float y, float theta)
{
    Eigen::Affine3f a = pcl::getTransformation(x, y, 0, 0, 0, theta);
    return a.matrix();
}

Eigen::Affine3f affine3fFrom(const Eigen::Matrix4f &m)
{
    Eigen::Affine3f a;
    a.matrix() = m;
    return a;
}

Eigen::Quaternionf quaternionfFrom(const Eigen::Matrix4f &m)
{
    return Eigen::Quaternionf(affine3fFrom(m).linear()).normalized();
}

Eigen::Matrix4f normalizeRotation(const Eigen::Matrix4f &m)
{
    Eigen::Affine3f a = affine3fFrom(m);
    a.linear() = Eigen::Quaternionf(a.linear()).normalized().toRotationMatrix();
    return a.matrix();
}

Eigen::Vector3f transformPoint(const Eigen::Vector3f &p, const Eigen::Matrix4f &m)
{
    Eigen::Vector3f out;
    pcl::transformPoint(p, out, affine3fFrom(m));
    return out;
}

Eigen::Vector3f pointFrom(const Eigen::Matrix4f &m)
{
    return m.topRightCorner(3, 1);
}

Eigen::Matrix4f rotationFrom(const Eigen::Matrix4f &m)
{
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
    out.topLeftCorner(3, 3) = m.topLeftCorner(3, 3);
    return out;
}

Eigen::Matrix4f translationFrom(const Eigen::Matrix4f &m)
{
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
    out.topRightCorner(3, 1) = m.topRightCorner(3, 1);
    return out;
}

Eigen::Vector4f vector4fZeroFrom(const Eigen::Matrix4f &m)
{
    return Eigen::Vector4f(m(0, 3), m(1, 3), m(2, 3), 0);
}

cv::Mat cvMatFrom(const Eigen::MatrixXf &m)
{
    cv::Mat mat(static_cast<int>(m.rows()), static_cast<int>(m.cols()), CV_32FC1);
    for (int r = 0; r < m.rows(); r++)
    {
        for (int c = 0; c < m.cols(); c++)
        {
            mat.at<float>(cv::Point(r, c)) = m(r, c);
        }
    }
    return mat;
}

QDebug operator<<(QDebug out, const cv::Mat &m)
{
    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            switch (m.type())
            {
            case CV_32F:
                out << fixed << qSetRealNumberPrecision(2) << qSetFieldWidth(6) << m.at<float>(cv::Point(j, i));
                break;
            case CV_16U:
                out << fixed << qSetRealNumberPrecision(2) << qSetFieldWidth(6) << m.at<ushort>(cv::Point(j, i));
                break;
            }
        }
        out << endl;
    }
    return out;
}

Eigen::Vector3f closedPointOnLine(const Eigen::Vector3f &point, const Eigen::Vector3f &dir, const Eigen::Vector3f &meanPoint)
{
    Eigen::Vector3f ev = point - meanPoint;
    Eigen::Vector3f closedPoint = meanPoint + dir * (ev.dot(dir));
    return closedPoint;
}

float oneAxisCoord(const Eigen::Vector3f& point, const Eigen::Vector3f& dir)
{
    Eigen::Vector3f projPt = dir * point.dot(dir);
    float coord = projPt.norm();
    if (projPt.dot(dir) < 0)
    {
        coord = -coord;
    }
    return coord;
}

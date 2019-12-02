#include "Device.h"

Device::Device(QObject *parent)
    : QObject(parent)
    , m_colorIntrinsic(Eigen::Matrix4f::Zero())
    , m_colorExtrinsic(Eigen::Matrix4f::Zero())
    , m_depthIntrinsic(Eigen::Matrix4f::Zero())
    , m_depthExtrinsic(Eigen::Matrix4f::Zero())
    , m_colorCompressionType(Frame::TYPE_COLOR_UNKNOWN)
    , m_depthCompressionType(Frame::TYPE_DEPTH_UNKNOWN)
    , m_colorSize(0, 0)
    , m_depthSize(0, 0)
    , m_depthShift(0)
{

}

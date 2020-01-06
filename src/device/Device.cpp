#include "Device.h"
#include "util/Utils.h"

#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <QDebug>

Device::Device(QObject *parent)
    : QObject(parent)
    , m_colorIntrinsic(Eigen::Matrix4f::Zero())
    , m_colorExtrinsic(Eigen::Matrix4f::Zero())
    , m_depthIntrinsic(Eigen::Matrix4f::Zero())
    , m_depthExtrinsic(Eigen::Matrix4f::Zero())
    , m_R(Eigen::Matrix3f::Zero())
    , m_colorSize(0, 0)
    , m_depthSize(0, 0)
    , m_depthShift(0)
{

}

void Device::initRectifyMap()
{
    cv::Mat cameraMatrix = cvMatFrom(m_colorIntrinsic.topLeftCorner(3, 3));
    cv::Mat distCoeffs;
    if (m_distCoeffs.size() < 4)
    {
        distCoeffs = cv::Mat(1, 4, CV_32F, cv::Scalar(0));
    }
    else
    {
        distCoeffs = cvMatFrom(m_distCoeffs);
    }
    cv::Size imgSize(m_colorSize.width(), m_colorSize.height());
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, 1, imgSize, 0);
    cv::Mat R;
    if (!m_R.isZero())
    {
        R = cvMatFrom(m_R);
    }
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, imgSize, CV_32F, m_rectifyMap1, m_rectifyMap2);
    //qDebug() << cameraMatrix;
    //qDebug() << distCoeffs;
    //qDebug() << newCameraMatrix;
    //qDebug() << R;
    //qDebug() << m_rectifyMap1;
    //qDebug() << m_rectifyMap2;
}

cv::Mat Device::undistortImage(const cv::Mat &in)
{
    cv::Mat out;
    cv::remap(in, out, m_rectifyMap1, m_rectifyMap2, cv::INTER_LINEAR);
    return out;
}

cv::Mat Device::alignDepthToColor(const cv::Mat &depthMat, const cv::Mat &colorMat)
{
    cv::Mat out;
    for (int r = 0; r < depthMat.rows; r++)
    {
        for (int c = 0; c < depthMat.cols; c++)
        {

        }
    }
    return out;
}

float Device::fx() const
{
    return m_colorIntrinsic(0, 0);
}

float Device::fy() const
{
    return m_colorIntrinsic(1, 1);
}

float Device::cx() const
{
    return m_colorIntrinsic(2, 0);
}

float Device::cy() const
{
    return m_colorIntrinsic(2, 1);
}

float Device::depthShift() const
{
    return m_depthShift;
}

//void alignFrame(const rs2_intrinsics& from_intrin,
//                                   const rs2_intrinsics& other_intrin,
//                                   rs2::frame from_image,
//                                   uint32_t output_image_bytes_per_pixel,
//                                   const rs2_extrinsics& from_to_other,
//								   cv::Mat & registeredDepth,
//								   float depth_scale_meters)
//{
//    static const auto meter_to_mm = 0.001f;
//    uint8_t* p_out_frame = registeredDepth.data;
//    auto from_vid_frame = from_image.as<rs2::video_frame>();
//    auto from_bytes_per_pixel = from_vid_frame.get_bytes_per_pixel();

//    static const auto blank_color = 0x00;
//    UASSERT(registeredDepth.total()*registeredDepth.channels()*registeredDepth.depth() == other_intrin.height * other_intrin.width * output_image_bytes_per_pixel);
//    memset(p_out_frame, blank_color, other_intrin.height * other_intrin.width * output_image_bytes_per_pixel);

//    auto p_from_frame = reinterpret_cast<const uint8_t*>(from_image.get_data());
//    auto from_stream_type = from_image.get_profile().stream_type();
//    float depth_units = ((from_stream_type == RS2_STREAM_DEPTH)? depth_scale_meters:1.f);
//    UASSERT(from_stream_type == RS2_STREAM_DEPTH);
//    UASSERT_MSG(depth_units > 0.0f, uFormat("depth_scale_meters=%f", depth_scale_meters).c_str());
//#pragma omp parallel for schedule(dynamic)
//    for (int from_y = 0; from_y < from_intrin.height; ++from_y)
//    {
//        int from_pixel_index = from_y * from_intrin.width;
//        for (int from_x = 0; from_x < from_intrin.width; ++from_x, ++from_pixel_index)
//        {
//            // Skip over depth pixels with the value of zero
//            float depth = (from_stream_type == RS2_STREAM_DEPTH)?(depth_units * ((const uint16_t*)p_from_frame)[from_pixel_index]): 1.f;
//            if (depth)
//            {
//                // Map the top-left corner of the depth pixel onto the other image
//                float from_pixel[2] = { from_x - 0.5f, from_y - 0.5f }, from_point[3], other_point[3], other_pixel[2];
//                rs2_deproject_pixel_to_point(from_point, &from_intrin, from_pixel, depth);
//                rs2_transform_point_to_point(other_point, &from_to_other, from_point);
//                rs2_project_point_to_pixel(other_pixel, &other_intrin, other_point);
//                const int other_x0 = static_cast<int>(other_pixel[0] + 0.5f);
//                const int other_y0 = static_cast<int>(other_pixel[1] + 0.5f);

//                // Map the bottom-right corner of the depth pixel onto the other image
//                from_pixel[0] = from_x + 0.5f; from_pixel[1] = from_y + 0.5f;
//                rs2_deproject_pixel_to_point(from_point, &from_intrin, from_pixel, depth);
//                rs2_transform_point_to_point(other_point, &from_to_other, from_point);
//                rs2_project_point_to_pixel(other_pixel, &other_intrin, other_point);
//                const int other_x1 = static_cast<int>(other_pixel[0] + 0.5f);
//                const int other_y1 = static_cast<int>(other_pixel[1] + 0.5f);

//                if (other_x0 < 0 || other_y0 < 0 || other_x1 >= other_intrin.width || other_y1 >= other_intrin.height)
//                    continue;

//                for (int y = other_y0; y <= other_y1; ++y)
//                {
//                    for (int x = other_x0; x <= other_x1; ++x)
//                    {
//                        int out_pixel_index = y * other_intrin.width + x;
//                        //Tranfer n-bit pixel to n-bit pixel
//                        for (int i = 0; i < from_bytes_per_pixel; i++)
//                        {
//                            const auto out_offset = out_pixel_index * output_image_bytes_per_pixel + i;
//                            const auto from_offset = from_pixel_index * output_image_bytes_per_pixel + i;
//                            p_out_frame[out_offset] = p_from_frame[from_offset] * (depth_units / meter_to_mm);
//                        }
//                    }
//                }
//            }
//        }
//    }
//}

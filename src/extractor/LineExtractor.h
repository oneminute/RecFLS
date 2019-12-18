#ifndef LINEEXTRACTOR_H
#define LINEEXTRACTOR_H

#include "LineSegment.h"

#include <QObject>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

template<typename PointT = pcl::PointXYZ>
struct PointComparator
{
    bool operator() (const PointT& v1, const PointT& v2)
    {
        if (v1.x < v2.x)
        {
            return true;
        }
        else if (v1.x == v2.x)
        {
            if (v1.y < v2.y)
            {
                return true;
            }
            else if (v1.y == v2.y)
            {
                if (v1.z < v2.z)
                    return true;
                else
                    return false;
            }
        }
        return false;
    }
};

template <typename PointInT, typename PointOutT>
class LineExtractor
{
    enum LINE_ORDER
    {
        LO_SS = 0,
        LO_SE,
        LO_ES,
        LO_EE
    };



public:
    LineExtractor()
        : segment_distance_threshold_(0.05), min_line_len_(9)
        , pca_error_threshold_(0.008f)
        , max_distance_between_two_lines_(0.05f)
        , max_error_(0.05f)
        , max_angle_error_(M_PI / 18)
        , nr_subdiv_(5)
        , pfh_tuple_()
        , d_pi_ (1.0f / (2.0f * static_cast<float> (M_PI)))
        , boundary_(new pcl::PointCloud<pcl::PointXYZI>)
        , lineCloud_(new pcl::PointCloud<pcl::PointXYZI>())
    {}

    void compute(const pcl::PointCloud<PointInT>& cloudIn, pcl::PointCloud<PointOutT>& cloudOut);

    std::map<int, int> linesCompare(const std::vector<LineSegment>& srcLines);

    std::vector<std::vector<int>> segments()
    {
        return segments_;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr getBoundary()
    {
        return boundary_;
    }

    std::vector<LineSegment> getLines()
    {
        return lines_;
    }

    std::vector<LineSegment> getMergedLines()
    {
        return mergedLines_;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr getLineCloud()
    {
        return lineCloud_;
    }

private:
//    void addPointToSortedIndices(const PointInT &pt);

    void joinSortedPoints();

    void extractLinesFromSegment(const std::vector<int>& segment, int segmentNo);

    void mergeCollinearLines();

    void mergeCollinearLines2();

    bool isLinesCollinear(const LineSegment &line1, const LineSegment &line2);

    bool isLinesCollinear2(const LineSegment &line1, const LineSegment &line2);

    float linesDistance(const LineSegment &line1, const LineSegment &line2, LINE_ORDER& order);

    float lineFit(const std::vector<int>& segment, int index, int length, Eigen::Vector3f& dir, Eigen::Vector3f& meanPoint);

    float distanceToLine(pcl::PointXYZI& point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint);

    float distanceToLine(Eigen::Vector3f point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint);

    Eigen::Vector3f closedPointOnLine(pcl::PointXYZI& point, Eigen::Vector3f& dir, Eigen::Vector3f meanPoint);

    void generateLineCloud();

private:
    std::vector<std::vector<int>> segments_;

    double segment_distance_threshold_;
    int min_line_len_;
    float pca_error_threshold_;
    float max_distance_between_two_lines_;
    float max_error_;
    float max_angle_error_;

    int nr_subdiv_;
    Eigen::Vector4f pfh_tuple_;
    int f_index_[3];
    float d_pi_;

    Eigen::Matrix3f intrinsic_;

    //pcl::PointCloud<PointInT>::Ptr downloadSampledCloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr boundary_;
    std::vector<LineSegment> lines_;
    std::vector<LineSegment> mergedLines_;

    std::map<pcl::PointXYZI, LineSegment, PointComparator<pcl::PointXYZI>> pointsLineMap_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr lineCloud_;
};

#endif // LINEEXTRACTOR_H

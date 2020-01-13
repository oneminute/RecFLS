#ifndef LINEEXTRACTOR_H
#define LINEEXTRACTOR_H

#include "LineCluster.h"
#include "LineSegment.h"
#include "LineTreeNode.h"

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

    enum LINE_RELATIONSHIP
    {
        LR_NONE = 0,
        LR_SIDE,
        LR_CHAIN_FW,
        LR_CHAIN_BW
    };

public:
    LineExtractor(
        float segmentDistanceThreshold = 0.1f,
        int minLinePoints = 9,
        float pcaErrorThreshold = 0.005f,
        float lineClusterAngleThreshold = 20.0f,
        float linesDistanceThreshold = 0.06f,
        float linesChainDistanceThreshold = 0.06f)
        : m_segmentDistanceThreshold(segmentDistanceThreshold)
        , m_minLinePoints(minLinePoints)
        , m_pcaErrorThreshold(pcaErrorThreshold)
        , m_lineClusterAngleThreshold(lineClusterAngleThreshold)
        , m_linesDistanceThreshold(linesDistanceThreshold)
        , m_linesChainDistanceThreshold(linesChainDistanceThreshold)
        , nr_subdiv_(5)
        , pfh_tuple_()
        , d_pi_ (1.0f / (2.0f * static_cast<float> (M_PI)))
        , m_boundary(new pcl::PointCloud<pcl::PointXYZI>)
        , m_lineCloud(new pcl::PointCloud<pcl::PointXYZI>())
        , m_root(nullptr)
    {}

    ~LineExtractor()
    {
        if (m_root)
        {
            m_root->deleteLater();
        }
    }

    bool LineCompare(const LineSegment& l1, const LineSegment& l2);

    void compute(const pcl::PointCloud<PointInT>& cloudIn, pcl::PointCloud<PointOutT>& cloudOut);

    std::map<int, int> linesCompare(const std::vector<LineSegment>& srcLines);

    std::vector<std::vector<int>> segments()
    {
        return m_segments;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr getBoundary()
    {
        return m_boundary;
    }

    std::vector<LineSegment> getLines()
    {
        return m_lines;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr getLineCloud()
    {
        return m_lineCloud;
    }

    std::vector<std::vector<int>> getSegments() const
    {
        return m_segments;
    }

    QList<LineCluster*> getLineClusters() const
    {
        return m_lineClusters;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr parameterizedLineMappingCluster();

    pcl::PointCloud<pcl::PointXYZI>::Ptr parameterizedPointMappingCluster(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloudIn, pcl::PointCloud<pcl::PointXYZ>::Ptr& dirCloud);

    QList<QList<int>> lineClusterFromParameterizedPointCloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

    QList<LineSegment> extractLinesFromClusters(const QList<QList<int>>& clusters, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

private:
    //    void addPointToSortedIndices(const PointInT &pt);

    void joinSortedPoints();

    void extractLinesFromSegment(const std::vector<int>& segment, int segmentNo);

    float lineFit(const std::vector<int>& segment, int index, int length, Eigen::Vector3f& dir, Eigen::Vector3f& meanPoint);

    float distanceToLine(pcl::PointXYZI& point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint);

    float distanceToLine(Eigen::Vector3f point, Eigen::Vector3f dir, Eigen::Vector3f meanPoint);

    Eigen::Vector3f closedPointOnLine(pcl::PointXYZI& point, Eigen::Vector3f& dir, Eigen::Vector3f meanPoint);

    void generateLineCloud();

    void unifyLineDirection(LineSegment &line);

    void linesSortingByLength(std::vector<LineSegment> &lines);

    void createLinesTree(const std::vector<LineSegment> &lines);

    void extracLinesClusters();

    void addLineTreeNode(LineTreeNode *node);

    void compareLines(LineSegment &longLine, LineSegment &shortLine, float &distance, float &angle, float &chainDistance, LINE_RELATIONSHIP &lineRel);

    void fetchSideNodes(LineTreeNode *node, std::vector<LineSegment> &cluster);

    LineTreeNode *findRightRoot(LineTreeNode *node);

    LineTreeNode *findLeftLeaf(LineTreeNode *node);

    void LineHoughCluster(float alpha, float beta, float distance);


private:
    std::vector<std::vector<int>> m_segments;

    float m_segmentDistanceThreshold;
    int m_minLinePoints;
    float m_pcaErrorThreshold;
    float m_lineClusterAngleThreshold;
    float m_linesDistanceThreshold;
    float m_linesChainDistanceThreshold;

    int nr_subdiv_;
    Eigen::Vector4f pfh_tuple_;
    int f_index_[3];
    float d_pi_;

    //pcl::PointCloud<PointInT>::Ptr downloadSampledCloud_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundary;
    std::vector<LineSegment> m_lines;
    QList<LineCluster*> m_lineClusters;

    std::map<pcl::PointXYZI, LineSegment, PointComparator<pcl::PointXYZI>> m_pointsLineMap;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_lineCloud;

    LineTreeNode *m_root;
};

#endif // LINEEXTRACTOR_H

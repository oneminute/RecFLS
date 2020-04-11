#ifndef LINEEXTRACTOR_H
#define LINEEXTRACTOR_H

#include <QObject>
#include <QList>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "common/Common.h"
#include "LineSegment.h"
#include "BoundaryExtractor.h"

#define LINE_MATCHER_DIVISION 4
#define LINE_MATCHER_DIST_DIVISION 6
#define LINE_MATCHER_ANGLE_ELEMDIMS (LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION)
#define LINE_MATCHER_ELEMDIMS (LINE_MATCHER_ANGLE_ELEMDIMS + LINE_MATCHER_DIST_DIVISION)

//struct MSLPoint
//{
//    union
//    {
//        float props[5];
//        struct
//        {
//            float alpha;
//            float beta;
//            float x;
//            float y;
//            float z;
//        };
//    };
//    static int propsSize() { return 5; }
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//};

struct PointLine
{
    union
    {
        float props[5];
        struct
        {
            float dAngleX;
            float dAngleY;
            float dist;
            float vAngleX;
            float vAngleY;
        };
    };
    static int propsSize() { return 5; }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <>
class pcl::DefaultPointRepresentation<PointLine> : public PointRepresentation<PointLine>
{
public:
    DefaultPointRepresentation()
    {
        nr_dimensions_ = PointLine::propsSize();
    }

    void copyToFloatArray(const PointLine& p, float* out) const override
    {
        for (int i = 0; i < nr_dimensions_; ++i)
            out[i] = p.props[i];
    }
};

struct Line
{
    Eigen::Vector3f dir;
    Eigen::Vector3f point;
    float weight;

    Eigen::Vector3f getEndPoint(float length)
    {
        return point + dir * length;
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

// ����
struct LineChain
{
    // ��ţ�������ͬ������ֱ�ߵ�����ͬ����������Ǹ�����ͬ����š�ʹ���ַ���������3-5��5-3����Ŷ�Ϊ"3_5"��
    QString group;

    // ��1��ֱ����ֱ�߼����е����
    int lineNo1;
    // ��2��ֱ����ֱ�߼����е����
    int lineNo2;

    // ��1��ֱ��
    Line line1;
    // ��2��ֱ��
    Line line2;

    // xLocal, yLocal, zLocal���������������ĵ�λ���������������һ����ά�ľֲ�����ϵ
    Eigen::Vector3f xLocal;
    Eigen::Vector3f yLocal;
    Eigen::Vector3f zLocal;

    // ֱ��1�ϵ������
    Eigen::Vector3f point1;
    // ֱ��2�ϵ������
    Eigen::Vector3f point2;
    // �����������е�
    Eigen::Vector3f point;

    // ��ֱ�ߵĽǶȣ��û��ȱ�ʾ
    float radians;
    // ��ֱ����̴��ߵĳ��ȣ�Ҳ��point1��point2�ľ���
    float length;

    // ����һ�����е�point��ƽ�棬��ƽ��������ֱ�߲�˺��������Ϊ�������ģ�����Ϊax + by + cz + d = 0
    pcl::ModelCoefficients::Ptr plane;

    QString name()
    {
        return QString("[%1 %2]").arg(lineNo1).arg(lineNo2);
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct LineDescriptor
{
    union
    {
        float props[5];
        struct
        {
            float x;
            float y;
            float z;
            float angleH;
            float angleV;
        };
    };
    static int propsSize() { return 5; }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template <>
class pcl::DefaultPointRepresentation<LineDescriptor> : public PointRepresentation<LineDescriptor>
{
public:
    DefaultPointRepresentation()
    {
        nr_dimensions_ = LineDescriptor::propsSize();
    }

    void copyToFloatArray(const LineDescriptor& p, float* out) const override
    {
        for (int i = 0; i < nr_dimensions_; ++i)
            out[i] = p.props[i];
    }
};

class LineExtractor : public QObject
{
    Q_OBJECT
public:
    enum ANGLE_MAPPING_METHOD
    {
        TWO_DIMS = 0,
        THREE_DIMS
    };
    
    explicit LineExtractor(QObject* parent = nullptr);

    QList<LineSegment> compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& boundaryCloud, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cornerCloud);

    void computeInternal(const pcl::PointCloud<pcl::PointXYZI>::Ptr& inCloud, 
                                    float radius, float a1dThreshold, 
                                    float lineInterval, float maxZdistance,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outCloud,
                                    pcl::IndicesPtr& outIndices,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outInlierCloud,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outOutlierCloud, 
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outAngleCloud,
                                    QList<int>& outAngleCloudIndices,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outDirCloud,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outMappingCloud,
                                    QList<float>& outDensityList,
                                    QList<float>& outOffsetRateList,
                                    pcl::PointCloud<Line>::Ptr& outMslCloud,
                                    pcl::PointCloud<PointLine>::Ptr& outLineCloud,
                                    QMap<int, std::vector<int>>& outSubCloudIndices,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outLinedCloud,
                                    QList<LineSegment>& outLineSegments,
                                    QList<float>& outErrors,
                                    QList<int>& outLinePointsCount);

    //void extractLinesFromPlanes(const QList<Plane>& planes);
    void groupLines(
        const QList<LineSegment>& inLineSegments,
        const pcl::PointCloud<PointLine>::Ptr& inLineCloud,
        float groupLinesSearchRadius,
        pcl::PointCloud<Line>::Ptr& outLineCloud,
        QList<LineSegment>& outLineSegments
    );

    void generateDescriptors();

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud() const
    {
        return m_cloud;
    }

    pcl::PointCloud<Line>::Ptr lineCloud() const
    {
        return m_lineCloud;
    }

    pcl::PointCloud<LineDescriptor>::Ptr descriptors() const { return m_descriptors; }
    //pcl::PointCloud<LineDescriptor2>::Ptr descriptors2() const { return m_descriptors2; }
    //pcl::PointCloud<LineDescriptor3>::Ptr descriptors3() const { return m_descriptors3; }

    cv::Mat board() const { return m_board; }
    void setBoard(const cv::Mat& board) { m_board = board; }

    QList<LineChain> chains() const { return m_chains; }

    Eigen::Vector3f lcLocalMiddle() const { return m_lcLocalMiddle; }

protected:

private:
    // �߽�����
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_cloud;

    pcl::PointCloud<Line>::Ptr m_lineCloud;

    QList<LineSegment> m_lineSegments;

    QList<LineChain> m_chains;

    pcl::PointCloud<LineDescriptor>::Ptr m_descriptors;
    //pcl::PointCloud<LineDescriptor2>::Ptr m_descriptors2;
    //pcl::PointCloud<LineDescriptor3>::Ptr m_descriptors3;

    cv::Mat m_board;

    ANGLE_MAPPING_METHOD m_angleMappingMethod;

    Eigen::Vector3f m_lcLocalMiddle;
    Eigen::Matrix4f m_lcLocalTransform;

    PROPERTY(float, BoundaryCloudA1dThreshold)
    PROPERTY(float, CornerCloudA1dThreshold)
    PROPERTY(float, BoundaryCloudSearchRadius)
    PROPERTY(float, CornerCloudSearchRadius)
    PROPERTY(float, PCASearchRadius)
    PROPERTY(int, MinNeighboursCount)
    PROPERTY(float, AngleCloudSearchRadius)
    PROPERTY(int, AngleCloudMinNeighboursCount)
    PROPERTY(float, MinLineLength)
    PROPERTY(float, BoundaryLineInterval)
    PROPERTY(float, CornerLineInterval)
    PROPERTY(float, BoundaryMaxZDistance)
    PROPERTY(float, CornerMaxZDistance)
    PROPERTY(float, CornerGroupLinesSearchRadius)

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // LINEEXTRACTOR_H
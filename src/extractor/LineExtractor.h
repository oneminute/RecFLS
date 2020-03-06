#ifndef LINEEXTRACTOR_H
#define LINEEXTRACTOR_H

#include <QObject>
#include <QList>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "LineSegment.h"
#include "BoundaryExtractor.h"

#define LINE_MATCHER_DIVISION 4
#define LINE_MATCHER_DIST_DIVISION 6
#define LINE_MATCHER_ANGLE_ELEMDIMS (LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION)
#define LINE_MATCHER_ELEMDIMS (LINE_MATCHER_ANGLE_ELEMDIMS + LINE_MATCHER_DIST_DIVISION)

struct MSLPoint
{
    union
    {
        float props[5];
        struct
        {
            float alpha;
            float beta;
            float x;
            float y;
            float z;
        };
    };
    static int propsSize() { return 5; }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

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
    LineDescriptor()
    {
        for (int i = 0; i < elemsSize(); i++)
        {
            elems[i] = 0;
        }
    }

    float elems[LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION];

    static int elemsSize() { return LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION; }
};

struct LineDescriptor2
{
    LineDescriptor2()
    {
        for (int i = 0; i < elemsSize(); i++)
        {
            elems[i] = 0;
        }
    }

    float elems[/*LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION + */ 11];

    static int elemsSize() { return /*LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION + */11; }
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

    QList<LineSegment> compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& boundaryCloud);

    void extractLinesFromPlanes(const QList<Plane>& planes);
    void segmentLines();

    void generateLineChains();

    bool generateLineChain(LineChain& lc);

    void generateDescriptors();
    void generateDescriptors2();
    void generateDescriptors3();

    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryCloud() const
    {
        return m_boundaryCloud;
    }

    pcl::IndicesPtr boundaryIndices() const
    {
        return m_boundaryIndices;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr angleCloud() const
    {
        return m_angleCloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr dirCloud() const
    {
        return m_dirCloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr mappingCloud() const
    {
        return m_mappingCloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr centerCloud() const
    {
        return m_centerCloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr linedCloud() const
    {
        return m_linedCloud;
    }

    //pcl::PointCloud<MSLPoint>::Ptr mslPointCloud() const
    //{
        //return m_mslPointCloud;
    //}

    pcl::PointCloud<Line>::Ptr mslCloud() const
    {
        return m_mslCloud;
    }

    QList<float> densityList() const
    {
        return m_density;
    }

    QList<int> angleCloudIndices() const
    {
        return m_angleCloudIndices;
    }

    QMap<int, std::vector<int>> subCloudIndices() const
    {
        return m_subCloudIndices;
    }

    QList<float> errors() const
    {
        return m_errors;
    }

    QList<int> linePointsCount() const
    {
        return m_linePointsCount;
    }

    pcl::PointCloud<LineDescriptor>::Ptr descriptors() const { return m_descriptors; }
    pcl::PointCloud<LineDescriptor2>::Ptr descriptors2() const { return m_descriptors2; }

    QList<LineChain> chains() const { return m_chains; }

    Eigen::Vector3f lcLocalMiddle() const { return m_lcLocalMiddle; }

    float boundBoxDiameter() const { return m_boundBoxDiameter; }

    float searchRadius() const { return m_searchRadius; }
    void setSearchRadius(float _searchRadius) { m_searchRadius = _searchRadius; }

    int minNeighbours() const { return m_minNeighbours; }
    void setMinNeighbours(int _minNeighbours) { m_minNeighbours = _minNeighbours; }

    float searchErrorThreshold() const { return m_searchErrorThreshold; }
    void setSearchErrorThreshold(float _searchErrorThreshold) { m_searchErrorThreshold = _searchErrorThreshold; }

    float angleSearchRadius() const { return m_angleSearchRadius; }
    void setAngleSearchRadius(float _angleSearchRadius) { m_angleSearchRadius = _angleSearchRadius; }

    int angleMinNeighbours() const { return m_angleMinNeighbours; }
    void setAngleMinNeighbours(int _angleMinNeighbours) { m_minNeighbours = _angleMinNeighbours; }

    float mappingTolerance() const { return m_mappingTolerance; }
    void setMappingTolerance(float _mappingTolerance) { m_mappingTolerance = _mappingTolerance; }

    float regionGrowingZDistanceThreshold() const { return m_regionGrowingZDistanceThreshold; }
    void setRegionGrowingZDistanceThreshold(float _regionGrowingZDistance) { m_regionGrowingZDistanceThreshold = _regionGrowingZDistance; }

    ANGLE_MAPPING_METHOD angleMappingMethod() const { return m_angleMappingMethod; }
    void setAngleMappingMethod(int _angleMappingMethod) { m_angleMappingMethod = static_cast<ANGLE_MAPPING_METHOD>(_angleMappingMethod); }

    float minLineLength() const { return m_minLineLength; }
    void setMinLineLength(float _minLineLength) { m_minLineLength = _minLineLength; }

    float mslRadiusSearch() const { return m_mslRadiusSearch; }
    void setMslRadiusSearch(float _mslRadiusSearch) { m_mslRadiusSearch = _mslRadiusSearch; }

protected:

private:
    // �߽�����
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryCloud;

    // �߽���������
    pcl::IndicesPtr m_boundaryIndices;

    QList<int> m_angleCloudIndices;

    // �߽�����������������
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_angleCloud;

    // �߽�����������
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_dirCloud;

    // ���������Ŀ���߶εĲ���������
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_mappingCloud;

    // �߽�����������������ÿ����Ľ�������
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_centerCloud;

    pcl::PointCloud<pcl::PointXYZI>::Ptr m_linedCloud;

    //pcl::PointCloud<MSLPoint>::Ptr m_mslPointCloud;

    pcl::PointCloud<PointLine>::Ptr m_lineCloud;

    pcl::PointCloud<Line>::Ptr m_mslCloud;

    QMap<int, std::vector<int>> m_subCloudIndices;

    // �߽�����������������ÿ�����ƫ����
    QList<float> m_offsetRate;

    // �߽�����������������ÿ����Ľ����ܶ�ֵ
    QList<float> m_density;

    QList<LineSegment> m_lineSegments;

    QList<float> m_errors;

    QList<int> m_linePointsCount;

    QList<LineChain> m_chains;

    pcl::PointCloud<LineDescriptor>::Ptr m_descriptors;
    pcl::PointCloud<LineDescriptor2>::Ptr m_descriptors2;

    ANGLE_MAPPING_METHOD m_angleMappingMethod;

    float m_searchRadius;
    int m_minNeighbours;
    float m_searchErrorThreshold;

    float m_angleSearchRadius;
    int m_angleMinNeighbours;

    float m_mappingTolerance;

    float m_regionGrowingZDistanceThreshold;

    float m_minLineLength;

    float m_mslRadiusSearch;

    Eigen::Vector3f m_maxPoint;
    Eigen::Vector3f m_minPoint;
    float m_boundBoxDiameter;

    float m_maxLineChainLength;

    Eigen::Vector3f m_lcLocalMiddle;
    Eigen::Matrix4f m_lcLocalTransform;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // LINEEXTRACTOR_H
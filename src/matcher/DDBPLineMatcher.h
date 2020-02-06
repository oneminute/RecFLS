#ifndef DDBPLINEMATCHER_H
#define DDBPLINEMATCHER_H

#include <QObject>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <extractor/DDBPLineExtractor.h>

#define LINE_MATCHER_DIVISION 4
#define LINE_MATCHER_DIST_DIVISION 6
#define LINE_MATCHER_ANGLE_ELEMDIMS (LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION)
#define LINE_MATCHER_ELEMDIMS (LINE_MATCHER_ANGLE_ELEMDIMS + LINE_MATCHER_DIST_DIVISION)

class DDBPLineMatcher : public QObject
{
    Q_OBJECT
public:
    struct LineChain
    {
        int line1;
        int line2;
        Eigen::Vector3f xLocal;
        Eigen::Vector3f yLocal;
        Eigen::Vector3f zLocal;
        Eigen::Vector3f point1;
        Eigen::Vector3f point2;

        QString name()
        {
            return QString("[%1 %2]").arg(line1).arg(line2);
        }
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

        float elems[LINE_MATCHER_ELEMDIMS];

        static int elemsSize() { return LINE_MATCHER_ELEMDIMS; }
    };

    explicit DDBPLineMatcher(QObject* parent = nullptr);

    Eigen::Matrix4f compute(
        pcl::PointCloud<MSLPoint>::Ptr firstPointCloud, 
        pcl::PointCloud<MSL>::Ptr firstLineCloud, 
        pcl::PointCloud<MSLPoint>::Ptr secondPointCloud,
        pcl::PointCloud<MSL>::Ptr secondLineCloud
    );

    Eigen::Quaternionf stepRotation(
        float firstDiameter,
        pcl::PointCloud<MSLPoint>::Ptr firstPointCloud, 
        pcl::PointCloud<MSL>::Ptr firstLineCloud, 
        float secondDiameter,
        pcl::PointCloud<MSLPoint>::Ptr secondPointCloud,
        pcl::PointCloud<MSL>::Ptr secondLineCloud,
        pcl::KdTreeFLANN<MSLPoint>::Ptr tree,
        float& rotationError,
        float& translationError,
        QMap<int, int>& pairs = QMap<int, int>()
    );

    Eigen::Vector3f stepTranslation(
        pcl::PointCloud<MSL>::Ptr firstLineCloud,
        pcl::PointCloud<MSL>::Ptr secondLineCloud,
        pcl::KdTreeFLANN<MSLPoint>::Ptr tree,
        float& translationError,
        QMap<int, int>& pairs = QMap<int, int>()
    );

protected:
    void generateDescriptors(pcl::PointCloud<MSL>::Ptr& lineCloud, 
    pcl::PointCloud<LineDescriptor>::Ptr& descriptors,
    QList<LineChain>& chains);

private:
    pcl::PointCloud<LineDescriptor>::Ptr m_descriptors1;
    pcl::PointCloud<LineDescriptor>::Ptr m_descriptors2;
    QList<LineChain> m_chains1;
    QList<LineChain> m_chains2;
    Eigen::MatrixXf m_descMat1;
    Eigen::MatrixXf m_descMat2;
};

#endif // DDBPLINEMATCHER_H

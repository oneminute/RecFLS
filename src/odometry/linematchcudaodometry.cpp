#include "LineMatchCudaOdometry.h"
#include "util/StopWatch.h"
#include "cuda/CudaInternal.h"
#include "device/Device.h"
#include "ui/CloudViewer.h"
//#include "extractor/LineExtractor.hpp"
#include "common/Parameters.h"

#include <QDebug>
#include <QFileDialog>

#include <pcl/common/common.h>
#include <pcl/features/boundary.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/filters/convolution.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/search/kdtree.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaimgproc.hpp>


// 曲线模型的顶点，模板参数：优化变量维度和数据类型
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        virtual void setToOriginImpl() // 重置
    {
        _estimate << 0, 0, 0;
    }

    virtual void oplusImpl(const double* update) // 更新
    {
        _estimate += Eigen::Vector3d(update);
    }
    // 存盘和读盘：留空
    virtual bool read(istream& in) { return true; }
    virtual bool write(ostream& out) const { return true; }
};

// 误差模型 模板参数：观测值维度，类型，连接顶点类型
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}
    // 计算曲线模型误差
    void computeError()
    {
        const CurveFittingVertex* v = static_cast<const CurveFittingVertex*> (_vertices[0]);
        const Eigen::Vector3d abc = v->estimate();
        _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
    }
    virtual bool read(istream& in) { return true; }
    virtual bool write(ostream& out) const { return true; }
public:
    double _x;  // x 值， y 值为 _measurement
};

bool LineMatchCudaOdometry::beforeProcessing(Frame& frame)
{
    if (!m_init)
    {
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 6>> Block;
        auto linearSolver = g2o::make_unique<g2o::LinearSolverDense<Block::PoseMatrixType>>();
        auto solverPtr = std::make_unique<Block>(std::move(linearSolver));
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solverPtr));
        m_optimizer.setAlgorithm(solver);
        m_optimizer.setVerbose(true);

        m_lineExtractor.reset(new FusedLineExtractor);
        m_lineMatcher.reset(new LineMatcher);

        m_dstFrame = m_lineExtractor->compute(frame);

        m_pose = Eigen::Matrix4f::Identity();
        m_dstFrame.setPose(m_pose);
        m_dstFrame.setKeyFrame();

        m_flFrames.insert(m_dstFrame.index(), m_dstFrame);

        m_init = true;
        return false;
    }
    return true;
}

void LineMatchCudaOdometry::saveCurrentFrame()
{
    QString fileName = QFileDialog::getSaveFileName(nullptr, tr("Save Boundaries"), QDir::currentPath(), tr("Polygon Files (*.obj *.ply *.pcd)"));
    qDebug() << "saving file" << fileName;
    //pcl::PointCloud<pcl::PointXYZ> cloud;
    //pcl::copyPointCloud(*m_boundaryCloud, cloud);
    //qDebug() << m_boundaryCloud->size() << cloud.size();
    //cloud.width = cloud.size();
    //cloud.height = 1;
    //cloud.is_dense = true;
    //pcl::io::savePCDFile<pcl::PointXYZ>(fileName.toStdString(), cloud);
}

void LineMatchCudaOdometry::optimize(FLFrame& prevFrame)
{
    //m_optimizer.clear();

    QList<qint64> indices;
    pcl::KdTreeFLANN<LineSegment>::Ptr tree(new pcl::KdTreeFLANN<LineSegment>());
    float error = 0;
    QMap<int, int> pairs;
    QMap<int, float> weights;
    QList<RelInformation::KeyPair> keys;
    keys.append(RelInformation::KeyPair(m_srcFrame.index(), m_srcFrame.prevIndex()));
    //if (m_flFrames.contains(prevFrame.prevIndex()))
    qint64 startIndex = m_srcFrame.index();
    qint64 endIndex = 0;
    for (int i = 0; i < 100; i++)
    {
        if (!m_flFrames.contains(prevFrame.prevIndex()))
        {
            endIndex = prevFrame.index();
            break;
        }

        keys.append(RelInformation::KeyPair(prevFrame.index(), prevFrame.prevIndex()));
        prevFrame = m_flFrames[prevFrame.prevIndex()];
        Eigen::Matrix4f prevPose = prevFrame.pose();
        Eigen::Matrix3f prevRot = prevPose.topLeftCorner(3, 3);
        Eigen::Vector3f prevTrans = prevPose.topRightCorner(3, 1);
        Eigen::Quaternionf prevQ(prevRot);

        Eigen::Matrix4f initPose = prevFrame.pose().inverse() * m_srcFrame.pose();
        tree->setInputCloud(prevFrame.lines());
        m_lineMatcher->match(m_srcFrame.lines(), m_dstFrame.lines(), tree, pairs, weights);
        Eigen::Matrix4f relPose = m_lineMatcher->step(m_srcFrame.lines(), prevFrame.lines(), initPose, error, pairs, weights);
        if (error >= 1)
            continue;

        /*RelInformation rel;
        rel.setKey(m_srcFrame.index(), prevFrame.index());
        rel.setTransform(relPose);
        rel.setError(error);
        m_relInfors.insert(rel.key(), rel);
        keys.append(rel.key());*/

        indices.append(prevFrame.index());
    }
    qDebug() << "prev fl frames:" << indices;

    //QMap<qint64, int> g2oMap;
    for (QList<RelInformation::KeyPair>::iterator i = keys.begin(); i != keys.end(); i++)
    {
        qint64 fromIndex = i->first;
        qint64 toIndex = i->second;
        RelInformation infor = m_relInfors[RelInformation::KeyPair(fromIndex, toIndex)];
        FLFrame from = m_flFrames[fromIndex];
        FLFrame to = m_flFrames[toIndex];

        if (m_optimizer.vertices().find(fromIndex) == m_optimizer.vertices().end())
        {
            g2o::VertexSE3* v = new g2o::VertexSE3();
            v->setId(fromIndex);
            g2o::Isometry3 t;
            t = from.rotation().cast<double>();
            t.translation() = from.translation().cast<double>();
            v->setEstimate(t);
            m_optimizer.addVertex(v);
            if (fromIndex == startIndex)
                v->setFixed(true);
        }
        if (m_optimizer.vertices().find(toIndex) == m_optimizer.vertices().end())
        {
            g2o::VertexSE3* v = new g2o::VertexSE3();
            v->setId(toIndex);
            g2o::Isometry3 t;
            t = to.rotation().cast<double>();
            t.translation() = to.translation().cast<double>();
            v->setEstimate(t);
            m_optimizer.addVertex(v);
            if (toIndex == endIndex)
                v->setFixed(true);
        }

        g2o::EdgeSE3* edge = new g2o::EdgeSE3();
        edge->setId(fromIndex * keys.size() + toIndex);
        edge->setVertex(0, m_optimizer.vertices()[fromIndex]);
        edge->setVertex(1, m_optimizer.vertices()[toIndex]);
        g2o::Isometry3 iso;
        iso = infor.rotationMatrix().cast<double>();
        iso.translation() = infor.translation().cast<double>();
        g2o::Vector7 meas = g2o::internal::toVectorQT(iso);
        g2o::Vector4::MapType(meas.data() + 3).normalize();
        edge->setMeasurement(g2o::internal::fromVectorQT(meas));
        m_optimizer.addEdge(edge);
    }
    m_optimizer.initializeOptimization();
    m_optimizer.optimize(100);

    for (g2o::HyperGraph::VertexIDMap::iterator i = m_optimizer.vertices().begin(); i != m_optimizer.vertices().end(); i++)
    {
        qint64 index = i->first;
        g2o::VertexSE3* vertex = reinterpret_cast<g2o::VertexSE3*>(i->second);
        g2o::Isometry3 t = vertex->estimate();
        FLFrame frame = m_flFrames[index];
        Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());
        pose.topLeftCorner(3, 3) = t.rotation().cast<float>();
        Eigen::Vector3f trans = t.translation().cast<float>();
        pose.topRightCorner(3, 1) = trans;
        frame.setPose(pose);
        m_poses[index] = pose;
    }

    for (g2o::HyperGraph::EdgeSet::iterator i = m_optimizer.edges().begin(); i != m_optimizer.edges().end(); i++)
    {
        g2o::EdgeSE3* edge = reinterpret_cast<g2o::EdgeSE3*>(*i);
        qint64 fromIndex = edge->vertex(0)->id();
        qint64 toIndex = edge->vertex(1)->id();

        g2o::Isometry3 iso = edge->measurement();
        Eigen::Matrix3f rot = iso.rotation().cast<float>();
        Eigen::Vector3f trans = iso.translation().cast<float>();
        Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());
        pose.topLeftCorner(3, 3) = rot;
        pose.topRightCorner(3, 1) = trans;

        RelInformation::KeyPair key(fromIndex, toIndex);
        m_relInfors[key].setTransform(pose);
    }
}

void LineMatchCudaOdometry::doProcessing(Frame& frame)
{
    m_optimizer.clear();

    m_srcFrame = m_lineExtractor->compute(frame);
	pcl::copyPointCloud(*m_lineExtractor->rgbCloud(), *m_cloud);

	//m_cloud = m_lineExtractor->cloud();
    m_srcFrame.setPrevIndex(m_dstFrame.index());
    Eigen::Matrix3f rot(Eigen::Matrix3f::Identity());
    Eigen::Vector3f trans(Eigen::Vector3f::Zero());
    Eigen::Matrix4f initPose(Eigen::Matrix4f::Identity());
    initPose.topLeftCorner(3, 3) = rot;
    initPose.topRightCorner(3, 1) = trans;

    m_srcFrame.setPose(initPose);
    pcl::KdTreeFLANN<LineSegment>::Ptr tree(new pcl::KdTreeFLANN<LineSegment>());
    tree->setInputCloud(m_dstFrame.lines());

    float error = 0;
    QMap<int, int> pairs;
    QMap<int, float> weights;
    m_lineMatcher->match(m_srcFrame.lines(), m_dstFrame.lines(), tree, pairs, weights);
    Eigen::Matrix4f poseDelta = m_lineMatcher->step(m_srcFrame.lines(), m_dstFrame.lines(), initPose, error, pairs, weights);
    if (error >= 1)
    {
        return;
    }
    RelInformation rel;
    rel.setKey(m_srcFrame.index(), m_dstFrame.index());
    rel.setTransform(poseDelta);
    rel.setError(error);
    m_relInfors.insert(rel.key(), rel);
    optimize(m_dstFrame);

    frame.setFrameIndex(m_poses.size());
    m_pose = poseDelta * m_pose;
    m_srcFrame.setPose(m_pose);
    FLFrame prevFrame = m_dstFrame;

    m_dstFrame = m_srcFrame;
    
    m_frames.append(frame);
    m_poses.insert(frame.frameIndex(), m_pose);
    m_flFrames.insert(m_dstFrame.index(), m_dstFrame);
}

void LineMatchCudaOdometry::afterProcessing(Frame& frame)
{
}

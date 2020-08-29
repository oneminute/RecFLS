#ifndef LINEMATCHCUDAODOMETRY_H
#define LINEMATCHCUDAODOMETRY_H

#include <QObject>

#include "Odometry.h"
#include "cuda/CudaInternal.h"

#include <pcl/common/common.h>
#include <pcl/gpu/containers/device_array.h>
#include <cuda_runtime.h>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/types/slam3d/vertex_se3.h>
#include <g2o/types/slam3d/edge_se3.h>
#include <g2o/types/slam3d/isometry3d_mappings.h>

#include "extractor/BoundaryExtractor.h"
#include "extractor/FusedLineExtractor.h"
#include "matcher/LineMatcher.h"
#include "device/SensorReaderDevice.h"
#include "extractor/LineSegment.h"
#include "common/RelInformation.h"

class LineMatchCudaOdometry : public Odometry
{
    Q_OBJECT
public:
    explicit LineMatchCudaOdometry(
        QObject* parent = nullptr)
        : Odometry(parent)
        , m_init(false)
    {}

    // Inherited via Odometry
    virtual void doProcessing(Frame& frame) override;
    virtual void afterProcessing(Frame& frame) override;
    virtual bool beforeProcessing(Frame& frame);
    virtual void saveCurrentFrame() override;

private:
    void optimize(FLFrame& prevFrame);

private:
    bool m_init;

    QScopedPointer<FusedLineExtractor> m_lineExtractor;
    QScopedPointer<LineMatcher> m_lineMatcher;

    FLFrame m_srcFrame;
    FLFrame m_dstFrame;
    QMap<int, FLFrame> m_flFrames;

    g2o::SparseOptimizer m_optimizer;
};

#endif // LINEMATCHCUDAODOMETRY_H

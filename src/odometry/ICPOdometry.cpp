#include "ICPOdometry.h"
#include "matcher/ICPMatcher.h"
#include "common/Parameters.h"

ICPOdometry::~ICPOdometry()
{
}

void ICPOdometry::doProcessing(Frame& frame)
{
    if (m_frameCount)
    {
        m_frameSrc.depthMatGpu.upload(frame.depthMat());
        cuda::icpGenerateCloud(m_frameSrc);
        m_cache.srcCloud = m_frameSrc.pointCloud;
        m_cache.srcNormals = m_frameSrc.pointCloudNormals;
        
        float error = 0;
        Eigen::Matrix3f m_rotation = Eigen::Matrix3f::Identity();
        Eigen::Vector3f m_translation = Eigen::Vector3f::Zero();
        Eigen::Matrix4f pose = m_icp->compute(m_cache, m_rotation, m_translation, error);
        m_pose = pose * m_pose;

        // swap src dst
        cuda::IcpFrame tmpFrame = m_frameDst;
        m_frameDst = m_frameSrc;
        m_frameSrc = tmpFrame;
        m_cache.swap();
    }
    else
    {
        m_frameDst.depthMatGpu.upload(frame.depthMat());
        cuda::icpGenerateCloud(m_frameDst);
        m_cache.dstCloud = m_frameDst.pointCloud;
        m_cache.dstNormals = m_frameDst.pointCloudNormals;
    }

    m_frames.append(frame);
    m_poses.append(m_pose);
    m_frameCount++;
}

void ICPOdometry::afterProcessing(Frame& frame)
{
}

bool ICPOdometry::beforeProcessing(Frame& frame)
{
    if (!m_init)
    {
        m_icp.reset(new ICPMatcher);

        cuda::IcpParameters parameters;
        parameters.cx = frame.getDevice()->cx();
        parameters.cy = frame.getDevice()->cy();
        parameters.fx = frame.getDevice()->fx();
        parameters.fy = frame.getDevice()->fy();
        parameters.minDepth = Settings::BoundaryExtractor_MinDepth.value();
        parameters.maxDepth = Settings::BoundaryExtractor_MaxDepth.value();
        parameters.depthShift = frame.getDevice()->depthShift();
        parameters.normalKernalRadius = Settings::ICPMatcher_CudaNormalKernalRadius.intValue();
        parameters.normalKnnRadius = Settings::ICPMatcher_CudaNormalKnnRadius.value();
        parameters.depthWidth = frame.getDepthWidth();
        parameters.depthHeight = frame.getDepthHeight();
        parameters.icpAnglesThreshold = Settings::ICPMatcher_AnglesThreshold.value();
        parameters.icpDistThreshold = Settings::ICPMatcher_DistanceThreshold.value();
        parameters.icpKernalRadius = Settings::ICPMatcher_IcpKernelRadius.intValue();
        parameters.blockSize = Settings::ICPMatcher_CudaBlockSize.intValue();

        m_frameSrc.parameters = parameters;
        m_frameDst.parameters = parameters;
        m_cache.parameters = parameters;

        m_frameSrc.allocate();
        m_frameDst.allocate();
        m_cache.allocate();

        m_pose = Eigen::Matrix4f::Identity();

        m_init = true;
    }
    return true;
}

void ICPOdometry::saveCurrentFrame()
{
}

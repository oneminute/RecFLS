#ifndef ICPODOMETRY_H
#define ICPODOMETRY_H

#include <QObject>

#include "Odometry.h"
#include "cuda/CudaInternal.h"

#include <pcl/common/common.h>
#include <pcl/gpu/containers/device_array.h>
#include <cuda_runtime.h>

#include "cuda/IcpInternal.h"
#include "device/SensorReaderDevice.h"
#include "matcher/ICPMatcher.h"

class ICPOdometry : public Odometry
{
    Q_OBJECT
public:
    explicit ICPOdometry(
        QObject* parent = nullptr)
        : Odometry(parent)
        , m_init(false)
        , m_frameCount(0)
    {}

    virtual ~ICPOdometry();

    // Inherited via Odometry
    virtual void doProcessing(Frame& frame) override;
    virtual void afterProcessing(Frame& frame) override;
    virtual bool beforeProcessing(Frame& frame);
    virtual void saveCurrentFrame() override;

private:
    cuda::IcpFrame m_frameSrc;
    cuda::IcpFrame m_frameDst;
    cuda::IcpCache m_cache;

    bool m_init;

    QScopedPointer<ICPMatcher> m_icp;
    int m_frameCount;
};

#endif // ICPODOMETRY_H
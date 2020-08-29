#include "SensorReaderDevice.h"
#include "common/Parameters.h"
#include "util/Utils.h"

#include <QDebug>
#include <QFile>
#include <QDataStream>

SensorReaderDevice::SensorReaderDevice()
    : m_frameCount(0)
    , m_imuFrameCount(0)
{
}


QString SensorReaderDevice::name() const
{
    return "SensorReader";
}

bool SensorReaderDevice::open()
{
    //QString fileName = Parameters::Global().stringValue("sample_path", "samples/office3.sens", "Device_SensorReader");
    QString fileName = Settings::SensorReader_SamplePath.value();
    QFile file(fileName);
    if (!file.exists())
    {
        qDebug() << "[SensorReaderDevice::open()]" << fileName << " is not exist.";
        return false;
    }

    if (!file.open(QIODevice::ReadOnly))
    {
        qDebug() << "[SensorReaderDevice::open()] Open " << fileName << " failure.";
        return false;
    }

    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::ByteOrder::LittleEndian);
    stream.setFloatingPointPrecision(QDataStream::SinglePrecision);

    // read version
    qint32 version = 0;
    stream >> version;

    // read sensor name
    quint64 sensorNameLength = 0;
    QByteArray sensorNameBytes;
    stream >> sensorNameLength;
    sensorNameBytes.resize(static_cast<int>(sensorNameLength));
    stream.readRawData(sensorNameBytes.data(), static_cast<int>(sensorNameLength));
    QString sensorName(sensorNameBytes);

    // read camera calibration parameters
    stream >> m_colorIntrinsic;

    for (int i = 0; i < m_colorExtrinsic.size(); i++)
    {
        float v = 0;
        stream >> v;
        m_colorExtrinsic.data()[i] = v;
    }
    for (int i = 0; i < m_depthIntrinsic.size(); i++)
    {
        float v = 0;
        stream >> v;
        m_depthIntrinsic.data()[i] = v;
    }
    for (int i = 0; i < m_depthExtrinsic.size(); i++)
    {
        float v = 0;
        stream >> v;
        m_depthExtrinsic.data()[i] = v;
    }

    // read compression type
    int colorCompressType = -1;
    int depthCompressType = -1;
    stream >> colorCompressType;
    stream >> depthCompressType;

    // read color and depth images' size
    int colorWidth, colorHeight;
    int depthWidth, depthHeight;

    stream >> colorWidth >> colorHeight;
    stream >> depthWidth >> depthHeight;
    stream >> m_depthShift;

    m_colorSize.setWidth(colorWidth);
    m_colorSize.setHeight(colorHeight);
    m_depthSize.setWidth(depthWidth);
    m_depthSize.setHeight(depthHeight);

    // read frame's count
    stream >> m_frameCount;
    for (quint64 i = 0; i < m_frameCount; i++)
    {
        Frame frame;
        frame.setDeviceFrameIndex(static_cast<int>(i));
        stream >> frame;
//        frame.showInfo();
        frame.setColorWidth(colorWidth);
        frame.setColorHeight(colorHeight);
        frame.setDepthWidth(depthWidth);
        frame.setDepthHeight(depthHeight);
        frame.setColorCompressionType(static_cast<Frame::COMPRESSION_TYPE_COLOR>(colorCompressType));
        frame.setDepthCompressionType(static_cast<Frame::COMPRESSION_TYPE_DEPTH>(depthCompressType));
        frame.setDevice(this);
        m_frames.append(frame);
    }

    stream >> m_imuFrameCount;
    for (quint64 i = 0; i < m_imuFrameCount; i++)
    {

    }

    file.close();
    qDebug() <<"[SensorReaderDevice::open()]" << "version =" << version;
    qDebug() <<"[SensorReaderDevice::open()]" << "sensor name =" << sensorName;
    qDebug() << "[SensorReaderDevice::open()]" << "color compression type = " << colorCompressType;
    qDebug() << "[SensorReaderDevice::open()]" << "depth compression type = " << depthCompressType;
    qDebug() << "[SensorReaderDevice::open()]" << "color size = " << m_colorSize;
    qDebug() << "[SensorReaderDevice::open()]" << "depth size = " << m_depthSize;
    qDebug() << "[SensorReaderDevice::open()]" << "depth shift = " << m_depthShift;
    qDebug() << "[SensorReaderDevice::open()]" << "frame count = " << m_frameCount;
    qDebug() << "[SensorReaderDevice::open()]" << "imu frame count = " << m_imuFrameCount;
    qDebug() << "[SensorReaderDevice::open()]" << "color intrincic =";
    qDebug() << m_colorIntrinsic;
    qDebug() << "[SensorReaderDevice::open()]" << "color extrinsic =";
    qDebug() << m_colorExtrinsic;
    qDebug() << "[SensorReaderDevice::open()]" << "depth intrincic =";
    qDebug() << m_depthIntrinsic;
    qDebug() << "[SensorReaderDevice::open()]" << "depth extrinsic =";
    qDebug() << m_depthExtrinsic;

    initRectifyMap();

    m_currentIndex = qMin<unsigned long long>(Settings::SensorReader_SkipFrames.intValue(), m_frameCount - 1);
    return true;
}

void SensorReaderDevice::close()
{
}

bool SensorReaderDevice::supportRandomAccessing()
{
    return true;
}

void SensorReaderDevice::skip(int skipCount)
{
    m_currentIndex = qMin<unsigned long long>(m_currentIndex + skipCount, m_frameCount - 1);
}

Frame SensorReaderDevice::getFrame(int frameIndex)
{
    Frame frame;
    if (frameIndex >= 0 && frameIndex < m_frameCount)
    {
        frame = m_frames[frameIndex];
        cv::Mat depthMat = frame.depthMat();
        qDebug() << depthMat.type() << depthMat.at<ushort>(100, 100);
    }
    return frame;
}

void SensorReaderDevice::fetchNext()
{
    if (m_currentIndex < m_frameCount)
    {
        emit frameFetched(getFrame(m_currentIndex++));
    }
    else
    {
        emit reachEnd();
    }
}

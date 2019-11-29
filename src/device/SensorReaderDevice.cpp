#include "SensorReaderDevice.h"
#include "common/Parameters.h"

#include <QDebug>
#include <QFile>
#include <QDataStream>

SensorReaderDevice::SensorReaderDevice()
{

}


QString SensorReaderDevice::name() const
{
    return "SensorReader";
}

bool SensorReaderDevice::open()
{
    QString fileName = Parameters::Global().stringValue("sample_path", "samples/office3.sens", "Device_SensorReader");
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
    for (int i = 0; i < m_colorIntrinsic.size(); i++)
    {
        float v = 0;
        stream >> v;
        m_colorIntrinsic.data()[i] = v;
    }
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

    m_colorCompressionType = static_cast<Frame::COMPRESSION_TYPE_COLOR>(colorCompressType);
    m_depthCompressionType = static_cast<Frame::COMPRESSION_TYPE_DEPTH>(depthCompressType);

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

    }

    file.close();
    qDebug() <<"[SensorReaderDevice::open()]" << "version =" << version;
    qDebug() <<"[SensorReaderDevice::open()]" << "sensor name =" << sensorName;
    qDebug() << "[SensorReaderDevice::open()]" << "color compression type = " << m_colorCompressionType;
    qDebug() << "[SensorReaderDevice::open()]" << "depth compression type = " << m_depthCompressionType;
    qDebug() << "[SensorReaderDevice::open()]" << "color size = " << m_colorSize;
    qDebug() << "[SensorReaderDevice::open()]" << "depth size = " << m_depthSize;
    qDebug() << "[SensorReaderDevice::open()]" << "depth shift = " << m_depthShift;
    qDebug() << "[SensorReaderDevice::open()]" << "frame count = " << m_frameCount;
    std::cout << "[SensorReaderDevice::open()] " << "color intrincic =" << std::endl;
    std::cout << m_colorIntrinsic << std::endl;
    std::cout << "[SensorReaderDevice::open()] " << "color extrinsic =" << std::endl;
    std::cout << m_colorExtrinsic << std::endl;
    std::cout << "[SensorReaderDevice::open()] " << "depth intrincic =" << std::endl;
    std::cout << m_depthIntrinsic << std::endl;
    std::cout << "[SensorReaderDevice::open()] " << "depth extrinsic =" << std::endl;
    std::cout << m_depthExtrinsic << std::endl;
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
}

Frame SensorReaderDevice::getFrame(int frameIndex)
{
    Frame frame;
    return frame;
}

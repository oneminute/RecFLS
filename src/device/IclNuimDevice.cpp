#include "IclNuimDevice.h"

#include "common/Parameters.h"
#include "util/Utils.h"

#include <QDir>
#include <QFileInfo>
#include <QTextStream>
#include <QFile>

#include <Eigen/Core>
#include <Eigen/Dense>

IclNuimDevice::IclNuimDevice(QObject* parent)
    : Device(parent)
{
    
}

QString IclNuimDevice::name() const
{
    return "IclNuimDevice";
}

bool IclNuimDevice::open()
{
    QDir baseDir(Settings::IclNuim_SamplePath.value());
    m_listFile = baseDir.absoluteFilePath(Settings::IclNuim_ListFile.value());
    QDir depthDir(baseDir.absoluteFilePath(Settings::IclNuim_DepthFolderName.value()));
    QDir rgbDir(baseDir.absoluteFilePath(Settings::IclNuim_RGBFolderName.value()));

    qDebug() << "List File:" << m_listFile;
    qDebug() << "Depth dir:" << depthDir;
    qDebug() << "RGB dir:" << rgbDir;

    QFile listFile(m_listFile);
    if (!listFile.exists())
    {
        qDebug() << m_listFile << "dos not exist.";
        return false;
    }

    listFile.open(QIODevice::ReadOnly | QIODevice::Text);
    QTextStream in(&listFile);
    QString line = in.readLine().trimmed();
    while (!line.isNull() && !line.isEmpty())
    {
        QStringList segs = line.split(" ");
        if (segs.length() == 4)
        {
            int index = segs[0].toInt();
            QString depthFile = baseDir.absoluteFilePath(segs[1]);
            QString rgbFile = baseDir.absoluteFilePath(segs[3]);
            //qDebug() << index << depthFile << rgbFile;

            m_frameIndices.append(index);
            m_depthFiles.append(depthFile);
            m_rgbFiles.append(rgbFile);
        }
        line = in.readLine().trimmed();
    }

    Eigen::Matrix4f intrinsic;
    intrinsic << 538.7, 0, 319.2, 0,
        0, 540.7, 233.6, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    Eigen::Matrix4f extrinsic(Eigen::Matrix4f::Identity());

    m_colorIntrinsic = m_depthIntrinsic = intrinsic.transpose();
    m_colorExtrinsic = m_depthExtrinsic = extrinsic;

    m_colorSize.setWidth(640);
    m_colorSize.setHeight(480);
    m_depthSize.setWidth(640);
    m_depthSize.setHeight(480);
    m_frameCount = m_frameIndices.size();
    m_depthShift = 1000;
    m_depthFactor = 5.0f;

    qDebug() << "[SensorReaderDevice::open()]" << "color size = " << m_colorSize;
    qDebug() << "[SensorReaderDevice::open()]" << "depth size = " << m_depthSize;
    qDebug() << "[SensorReaderDevice::open()]" << "depth shift = " << m_depthShift;
    qDebug() << "[SensorReaderDevice::open()]" << "frame count = " << m_frameCount;
    qDebug() << "[SensorReaderDevice::open()]" << "color intrincic =";
    qDebug() << m_colorIntrinsic;
    qDebug() << "[SensorReaderDevice::open()]" << "color extrinsic =";
    qDebug() << m_colorExtrinsic;
    qDebug() << "[SensorReaderDevice::open()]" << "depth intrincic =";
    qDebug() << m_depthIntrinsic;
    qDebug() << "[SensorReaderDevice::open()]" << "depth extrinsic =";
    qDebug() << m_depthExtrinsic;

    initRectifyMap();

    m_currentIndex = 0;

    return true;
}

void IclNuimDevice::close()
{
}

bool IclNuimDevice::supportRandomAccessing()
{
    return false;
}

void IclNuimDevice::skip(int skipCount)
{
    m_currentIndex = qMin<unsigned long long>(m_currentIndex + skipCount, m_frameCount - 1);
}

Frame IclNuimDevice::getFrame(int frameIndex)
{
    Frame frame;
    int index = m_frameIndices[frameIndex];
    QString depthFile = m_depthFiles[index];
    QString rgbFile = m_rgbFiles[index];
    cv::Mat depthMat = cv::imread(depthFile.toStdString(), cv::IMREAD_UNCHANGED);
    cv::Mat rgbMat = cv::imread(rgbFile.toStdString(), cv::IMREAD_UNCHANGED);

    //qDebug() << depthMat.type() << depthMat.at<ushort>(100, 100);

    frame.setDepthMat(depthMat);
    frame.setColorMat(rgbMat);

    frame.setColorWidth(m_colorSize.width());
    frame.setColorHeight(m_colorSize.height());
    frame.setDepthWidth(m_depthSize.width());
    frame.setDepthHeight(m_depthSize.height());

    frame.setDevice(this);
    frame.setDeviceFrameIndex(index);
    frame.setFrameIndex(index);

    return frame;
}

Frame IclNuimDevice::fetchNext()
{
    Frame frame;
	qDebug() << "m_currentIndex =" << m_currentIndex;
    if (m_currentIndex < m_frameCount)
    {
		frame = getFrame(m_currentIndex);
        emit frameFetched(frame);
		m_currentIndex++;
    }
    else
    {
        emit reachEnd();
    }
	return frame;
}

void IclNuimDevice::start()
{
}

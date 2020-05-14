#include "Parameters.h"

#include <QFile>
#include <QDebug>
#include <QMutexLocker>

QMap<QString, SettingItem*> SettingItem::m_items;

//IMPLEMENT_RANGE_SETTING(KNNRadius, 0.1f, 0.1f, 0.005f, 1.f, 0.005f, BoundaryExtractor, "KNN search radius.");
IMPLEMENT_RANGE_SETTING(BorderLeft, 26, 26, 0, 320, 1, BoundaryExtractor, "Pixels to left border.");
IMPLEMENT_RANGE_SETTING(BorderRight, 8, 8, 0, 320, 1, BoundaryExtractor, "Pixels to right border.");
IMPLEMENT_RANGE_SETTING(BorderTop, 4, 4, 0, 240, 1, BoundaryExtractor, "Pixels to top border.");
IMPLEMENT_RANGE_SETTING(BorderBottom, 4, 4, 0, 240, 1, BoundaryExtractor, "Pixels to bottom border.");
IMPLEMENT_RANGE_SETTING(MinDepth, 0.4f, 0.4f, 0, 5.0f, 0.1f, BoundaryExtractor, "Min depth threshold.");
IMPLEMENT_RANGE_SETTING(MaxDepth, 8.0f, 8.0f, 0, 10000.0f, 0.5f, BoundaryExtractor, "Max depth threshold.");
IMPLEMENT_RANGE_SETTING(CudaNormalKernalRadius, 20, 20, 5, 40, 1, BoundaryExtractor, "Normal estimation kernal radius.");
IMPLEMENT_RANGE_SETTING(CudaNormalKnnRadius, 0.1, 0.1, 0.01, 1, 0.01, BoundaryExtractor, "Normal estimation knn radius.");
IMPLEMENT_RANGE_SETTING(CudaBEDistance, 0.1f, 0.1f, 0.01, 1, 0.01, BoundaryExtractor, "Cuda boundary estimation distance.");
IMPLEMENT_RANGE_SETTING(CudaBEAngleThreshold, 45, 45, 45, 180, 5, BoundaryExtractor, "Cuda boundary estimation max angle threshold.");
IMPLEMENT_RANGE_SETTING(CudaBEKernalRadius, 20, 20, 5, 50, 1, BoundaryExtractor, "Cuda boundary estimation kernal radius.");
IMPLEMENT_RANGE_SETTING(CudaGaussianSigma, 4.0f, 4.0f, 0.1f, 40.0f, 0.5f, BoundaryExtractor, "Cuda gaussian sigma radius.");
IMPLEMENT_RANGE_SETTING(CudaGaussianKernalRadius, 20, 20, 5, 50, 1, BoundaryExtractor, "Cuda gaussian kernal radius.");
IMPLEMENT_RANGE_SETTING(CudaClassifyKernalRadius, 20, 20, 5, 50, 1, BoundaryExtractor, "Cuda classify kernal radius.");
IMPLEMENT_RANGE_SETTING(CudaClassifyDistance, 0.2f, 0.2f, 0.1f, 1.0f, 0.01f, BoundaryExtractor, "Cuda classify distance.");
IMPLEMENT_RANGE_SETTING(CudaPeakClusterTolerance, 5, 5, 1, 90, 1, BoundaryExtractor, "Cuda peak cluster tolerance.");
IMPLEMENT_RANGE_SETTING(CudaMinClusterPeaks, 2, 2, 1, 5, 1, BoundaryExtractor, "Cuda min cluster peaks.");
IMPLEMENT_RANGE_SETTING(CudaMaxClusterPeaks, 3, 3, 1, 5, 1, BoundaryExtractor, "Cuda max cluster peaks.");
IMPLEMENT_RANGE_SETTING(CudaCornerHistSigma, 1.0f, 1.0f, 0, 10.f, 0.1, BoundaryExtractor, "Cuda corner hist sigma.");

IMPLEMENT_RANGE_SETTING(BoundaryCloudA1dThreshold, 4.0f, 4.0f, 1.0f, 100.f, 1.0f, LineExtractor, "Using for a1d judgement.");
IMPLEMENT_RANGE_SETTING(CornerCloudA1dThreshold, 3.0f, 3.0f, 1.0f, 100.f, 1.0f, LineExtractor, "Using for a1d judgement.");
IMPLEMENT_RANGE_SETTING(BoundaryCloudSearchRadius, 0.1f, 0.1f, 0.01f, 1.f, 0.1f, LineExtractor, "Boundary cloud search radius.");
IMPLEMENT_RANGE_SETTING(CornerCloudSearchRadius, 0.2f, 0.2f, 0.01f, 1.f, 0.1f, LineExtractor, "Corner cloud search radius.");
IMPLEMENT_RANGE_SETTING(PCASearchRadius, 0.1f, 0.1f, 0.01f, 1.f, 0.01f, LineExtractor, "PCA search radius.");
IMPLEMENT_RANGE_SETTING(MinNeighboursCount, 10, 10, 3, 5000, 5, LineExtractor, "Min neighbours count.");
IMPLEMENT_RANGE_SETTING(AngleCloudSearchRadius, 20, 20, 1, 90, 1, LineExtractor, "Angle cloud search radius.");
IMPLEMENT_RANGE_SETTING(AngleCloudMinNeighboursCount, 10, 10, 3, 5000, 1, LineExtractor, "Angle cloud min neighbours count.");
IMPLEMENT_RANGE_SETTING(MinLineLength, 0.1f, 0.1f, 0.01f, 1, 0.01f, LineExtractor, "Min line length.");
IMPLEMENT_RANGE_SETTING(BoundaryLineInterval, 0.1f, 0.1f, 0.01f, 1.f, 0.05f, LineExtractor, "");
IMPLEMENT_RANGE_SETTING(CornerLineInterval, 0.2f, 0.2f, 0.01f, 1.f, 0.05f, LineExtractor, "");
IMPLEMENT_RANGE_SETTING(BoundaryMaxZDistance, 0.01f, 0.01f, 0.001f, 1.f, 0.01f, LineExtractor, "");
IMPLEMENT_RANGE_SETTING(CornerMaxZDistance, 0.05f, 0.05f, 0.001f, 1.f, 0.01f, LineExtractor, "");
IMPLEMENT_RANGE_SETTING(BoundaryGroupLinesSearchRadius, 0.05f, 0.05f, 0.001f, 3.f, 0.01f, LineExtractor, "");
IMPLEMENT_RANGE_SETTING(CornerGroupLinesSearchRadius, 0.05f, 0.05f, 0.001f, 3.f, 0.01f, LineExtractor, "");

IMPLEMENT_RANGE_SETTING(MaxIterations, 30, 30, 1, 100, 1, LineMatcher, "Max iterations.");

IMPLEMENT_RANGE_SETTING(AnglesThreshold, 0.95f, 0.95f, 0.9f, 1.f, 0.01f, ICPMatcher, "");
IMPLEMENT_RANGE_SETTING(DistanceThreshold, 0.05f, 0.05f, 0.01f, 0.2f, 0.01f, ICPMatcher, "");
IMPLEMENT_RANGE_SETTING(IcpKernelRadius, 5, 5, 1, 20, 1, ICPMatcher, "");
IMPLEMENT_RANGE_SETTING(CudaBlockSize, 32, 32, 32, 512, 16, ICPMatcher, "");
IMPLEMENT_RANGE_SETTING(CudaNormalKernalRadius, 10, 10, 5, 40, 1, ICPMatcher, "Normal estimation kernal radius.");
IMPLEMENT_RANGE_SETTING(CudaNormalKnnRadius, 0.1, 0.1, 0.01, 1, 0.01, ICPMatcher, "Normal estimation knn radius.");
IMPLEMENT_RANGE_SETTING(MaxIterations, 20, 20, 1, 1000, 1, ICPMatcher, "Max iterations.");

IMPLEMENT_STRING_SETTING(SamplePath, "samples/office3.sens", "samples/office3.sens", SensorReader, "Sample dataset's file path.");
IMPLEMENT_RANGE_SETTING(SkipFrames, 0, 0, 0, 65535, 10, SensorReader, "Frames Skipped.");

void Settings::save()
{
    QSettings settings("config2.ini", QSettings::IniFormat);
    for (SettingItem::SettingMap::iterator i = SettingItem::items().begin(); i != SettingItem::items().end(); i++)
    {
        QString fullKey = i.key();
        SettingItem* setting = i.value();

        QVariant value = setting->serialize();

        setting->debugPrint();

        settings.setValue(fullKey, value.toString());
    }
}

void Settings::load()
{
    QSettings settings("config2.ini", QSettings::IniFormat);
    for (SettingItem::SettingMap::iterator i = SettingItem::items().begin(); i != SettingItem::items().end(); i++)
    {
        QString fullKey = i.key();
        SettingItem* setting = i.value();

        if (settings.contains(fullKey))
        {
            setting->deserialize(settings.value(fullKey));
        }
        else
        {
            setting->restore();
        }

        setting->debugPrint();
    }
}

void Settings::restore()
{
    for (SettingItem::SettingMap::iterator i = SettingItem::items().begin(); i != SettingItem::items().end(); i++)
    {
        QString fullKey = i.key();
        SettingItem* setting = i.value();
        setting->restore();
    }
}


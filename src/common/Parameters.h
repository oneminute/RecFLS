#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <QtMath>
#include <QDebug>
#include <QMap>
#include <QList>
#include <QObject>
#include <QScopedPointer>
#include <QSettings>
#include <QThread>
#include <QMutex>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QLabel>
#include <QLineEdit>
#include <QComboBox>

class SettingItem
{
public:
    typedef QMap<QString, SettingItem*> SettingMap;

    SettingItem(QString key, const QString& group = "General", const QString& description = "")
        : m_key(key)
        , m_group(group)
        , m_description(description)
        , m_widget(nullptr)
    {
        m_items.insert(fullKey(), this);
    }

    QString key() const
    {
        return m_key;
    }

    QString group() const
    {
        return m_group;
    }

    QString fullKey() const
    {
        if (m_group == "General")
        {
            return m_key;
        }
        else
        {
            return QString("%1/%2").arg(m_group).arg(m_key);
        }
    }

    QString description() const
    {
        return m_description;
    }

    static SettingMap& items()
    {
        return m_items;
    }

    virtual QVariant serialize() = 0;
    virtual void deserialize(const QVariant& value) = 0;
    virtual void restore() = 0;
    virtual QWidget* createWidget() = 0;
    
    virtual void debugPrint()
    {
        qDebug().nospace().noquote() << "[" << "key = " << m_key << ", group = " << m_group << ", description = " << m_description << "]";
    }

protected:
    QString m_key;
    QString m_group;
    QString m_description;
    QWidget* m_widget;

private:
    static SettingMap m_items;
};

template<class T>
class BaseSetting: public SettingItem
{
public:
    BaseSetting(
        const QString& key,
        T value,
        T defaultValue,
        const QString& group = "General",
        const QString& description = ""
    )
        : SettingItem(key, group, description)
        , m_value(value)
        , m_defaultValue(defaultValue)
    {}

    T value()
    {
        return m_value;
    }

    T defaultValue()
    {
        return m_defaultValue;
    }

    void setValue(T _value)
    {
        m_value = _value;
    }

    virtual void restore()
    {
        m_value = m_defaultValue;
    }

    virtual QVariant serialize()
    {
        return QVariant(m_value);
    }

    virtual void debugPrint()
    {
        qDebug().nospace().noquote() << "[" << "key = " << m_key << ", value = " << m_value 
                                     << ", defaultValue = " << m_defaultValue << ", group = " << m_group 
                                     << ", description = " << m_description << "]";
    }

protected:
    T m_value;
    T m_defaultValue;
};

class RangeSetting : public BaseSetting<float>
{
public:
    explicit RangeSetting(const QString& key, float value, float defaultValue, float min, float max, float step, const QString& group = "General", const QString& description = "")
        : BaseSetting<float>(key, value, defaultValue, group, description)
        , m_min(min)
        , m_max(max)
        , m_step(step)
    {}

    virtual void deserialize(const QVariant& value)
    {
        m_value = value.toFloat();
    }

    virtual QWidget* createWidget()
    {
        QDoubleSpinBox* widget = new QDoubleSpinBox;
        widget->setDecimals(3);
        widget->setMinimum(m_min);
        widget->setMaximum(m_max);
        widget->setSingleStep(m_step);
        widget->setValue(static_cast<double>(m_value));
        QObject::connect(widget, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [this](double value) -> void
            {
                this->setValue(value);
            }
        );
        return widget;
    }
    
    virtual int intValue()
    {
        return static_cast<int>(m_value);
    }

private:
    float m_min;
    float m_max;
    float m_step;
};

class IntSetting : public BaseSetting<int>
{
public:
    explicit IntSetting(const QString& key, int value, int defaultValue, const QString& group = "General", const QString& description = "")
        : BaseSetting<int>(key, value, defaultValue, group, description)
    {}

    virtual void deserialize(const QVariant& value)
    {
        m_value = value.toInt();
    }

    virtual QWidget* createWidget()
    {
        QSpinBox* widget = new QSpinBox;
        
        widget->setValue(m_value);
        return widget;
    }
};

class StringSetting : public BaseSetting<QString>
{
public:
    explicit StringSetting(const QString& key, const QString& value, const QString& defaultValue, const QString& group = "General", const QString& description = "")
        : BaseSetting<QString>(key, value, defaultValue, group, description)
    {}

    virtual void deserialize(const QVariant& value)
    {
        m_value = value.toString();
    }

    virtual QWidget* createWidget()
    {
        QLineEdit* widget = new QLineEdit();
        widget->setText(m_value);
        return widget;
    }
};

#define DECLARE_SETTING(settingType, key, group) \
    public: \
        static settingType group##_##key

#define IMPLEMENT_RANGE_SETTING(key, value, defaultValue, min, max, step, group, description) \
    RangeSetting Settings::group##_##key(#key, value, defaultValue, min, max, step, #group, tr(description))

#define IMPLEMENT_STRING_SETTING(key, value, defaultValue, group, description) \
    StringSetting Settings::group##_##key(#key, value, defaultValue, #group, description)

class Settings : public QObject
{
    Q_OBJECT
public:
    typedef QMap<QString, QVariant> SettingMap;

    explicit Settings(QObject* parent = nullptr);

    static void save();
    static void load();
    static void restore();

    DECLARE_SETTING(RangeSetting, BorderLeft, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, BorderRight, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, BorderTop, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, BorderBottom, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, MinDepth, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, MaxDepth, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaNormalKernalRadius, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaNormalKnnRadius, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaBEDistance, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaBEAngleThreshold, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaBEKernalRadius, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaGaussianSigma, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaGaussianKernalRadius, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaClassifyKernalRadius, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaClassifyDistance, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaPeakClusterTolerance, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaMinClusterPeaks, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaMaxClusterPeaks, BoundaryExtractor);
    DECLARE_SETTING(RangeSetting, CudaCornerHistSigma, BoundaryExtractor);

    DECLARE_SETTING(RangeSetting, BoundaryCloudA1dThreshold, LineExtractor);
    DECLARE_SETTING(RangeSetting, CornerCloudA1dThreshold, LineExtractor);
    DECLARE_SETTING(RangeSetting, BoundaryCloudSearchRadius, LineExtractor);
    DECLARE_SETTING(RangeSetting, CornerCloudSearchRadius, LineExtractor);
    DECLARE_SETTING(RangeSetting, PCASearchRadius, LineExtractor);
    DECLARE_SETTING(RangeSetting, MinNeighboursCount, LineExtractor);
    DECLARE_SETTING(RangeSetting, AngleCloudSearchRadius, LineExtractor);
    DECLARE_SETTING(RangeSetting, AngleCloudMinNeighboursCount, LineExtractor);
    DECLARE_SETTING(RangeSetting, MinLineLength, LineExtractor);
    DECLARE_SETTING(RangeSetting, BoundaryLineInterval, LineExtractor);
    DECLARE_SETTING(RangeSetting, CornerLineInterval, LineExtractor);
    DECLARE_SETTING(RangeSetting, BoundaryMaxZDistance, LineExtractor);
    DECLARE_SETTING(RangeSetting, CornerMaxZDistance, LineExtractor);
    DECLARE_SETTING(RangeSetting, BoundaryGroupLinesSearchRadius, LineExtractor);
    DECLARE_SETTING(RangeSetting, CornerGroupLinesSearchRadius, LineExtractor);

    DECLARE_SETTING(RangeSetting, MaxIterations, LineMatcher);

    DECLARE_SETTING(RangeSetting, AnglesThreshold, ICPMatcher);
    DECLARE_SETTING(RangeSetting, DistanceThreshold, ICPMatcher);
    DECLARE_SETTING(RangeSetting, IcpKernelRadius, ICPMatcher);
    DECLARE_SETTING(RangeSetting, CudaBlockSize, ICPMatcher);
    DECLARE_SETTING(RangeSetting, CudaNormalKernalRadius, ICPMatcher);
    DECLARE_SETTING(RangeSetting, CudaNormalKnnRadius, ICPMatcher);
    DECLARE_SETTING(RangeSetting, MaxIterations, ICPMatcher);

    DECLARE_SETTING(StringSetting, SamplePath, SensorReader);
    DECLARE_SETTING(RangeSetting, SkipFrames, SensorReader);

    DECLARE_SETTING(StringSetting, SamplePath, IclNuim);
    DECLARE_SETTING(StringSetting, ListFile, IclNuim);
    DECLARE_SETTING(StringSetting, DepthFolderName, IclNuim);
    DECLARE_SETTING(StringSetting, RGBFolderName, IclNuim);
    DECLARE_SETTING(StringSetting, PosesFile, IclNuim);

    DECLARE_SETTING(StringSetting, DeviceName, Device);

private:
    SettingMap m_settings;
    SettingMap m_defaultSettings;
    bool m_modified;
};


#endif // PARAMETERS_H

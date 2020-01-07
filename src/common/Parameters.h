#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <QMap>
#include <QObject>
#include <QScopedPointer>
#include <QSettings>
#include <QThread>
#include <QMutex>

#define DEFAULT_SETTINGS_GROUP "General"
#define PARAMETERS Parameters::Global()

class ParameterWriter : public QObject
{
    Q_OBJECT
public:
    explicit ParameterWriter(QObject* parent = nullptr)
        : QObject(parent)
        , m_settings(nullptr)
    {}

public slots:
    void setValue(const QString& key, const QVariant& value)
    {
        Q_ASSERT(m_settings);
        QMutexLocker locker(&m_writerMutex);
        m_settings->setValue(key, value.toString());
    }

    void setValues(const QMap<QString, QVariant> values)
    {
        Q_ASSERT(m_settings);
        QMutexLocker locker(&m_writerMutex);
        for (QMap<QString, QVariant>::const_iterator i = values.begin(); i != values.end(); i++)
        {
            m_settings->setValue(i.key(), i.value().toString());
        }
    }

    void setSettings(QSettings* settings)
    {
        m_settings = settings;
    }

private:
    QSettings* m_settings;
    QMutex m_writerMutex;
};

class Parameters : public QObject
{
    Q_OBJECT
public:
    explicit Parameters(QObject *parent = nullptr);

    ~Parameters();

    static Parameters& Global();

    void load(const QString &path = "config.ini");

    QString stringValue(const QString &key, const QString &defaultValue="", const QString &group = "General");

    bool boolValue(const QString &key, bool defaultValue = false, const QString &group = "General");

    int intValue(const QString &key, int defaultValue = 0, const QString &group = "General");

    float floatValue(const QString &key, float defaultValue = 0.0f, const QString &group = "General");

    void setValue(const QString& key, const QVariant& value, const QString &group = "General");

    QVariant value(const QString& key, const QVariant& value);

    void save();

    // [General] settings
    bool debugMode();
    void setDebugMode(bool value);

    QString version();
    void setVersion(const QString& value);
    // End [General] settings

signals:
    void setValueSignal(const QString& key, const QVariant& value);
    void setValuesSignal(const QMap<QString, QVariant> &values);

public slots:

private:
    QString getFullKey(const QString& key, const QString& group);

private:
    QScopedPointer<QSettings> m_settings;
    QMap<QString, QVariant> m_cache;
    ParameterWriter* m_writer;
    QThread m_writerThread;
    QMutex m_cacheMutex;
};

#endif // PARAMETERS_H

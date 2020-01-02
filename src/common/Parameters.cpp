#include "Parameters.h"

#include <QFile>
#include <QDebug>
#include <QMutexLocker>

#define BEGIN_CHECK_GROUP(group) \
    if (group != DEFAULT_SETTINGS_GROUP) \
        m_settings->beginGroup(group)

#define CHECK_SETTING(key, defaultValue) \
    if (!m_settings->contains(key)) \
        setValue(key, defaultValue)

#define END_CHECK_GROUP(group) \
    if (group != DEFAULT_SETTINGS_GROUP) \
        m_settings->endGroup()

Parameters::Parameters(QObject *parent) 
    : QObject(parent)
    , m_settings(nullptr)
    , m_writer(new ParameterWriter)
{
    m_writer->moveToThread(&m_writerThread);
    connect(&m_writerThread, &QThread::finished, m_writer, &QObject::deleteLater);
    connect(this, &Parameters::setValueSignal, m_writer, &ParameterWriter::setValue, Qt::QueuedConnection);
    connect(this, &Parameters::setValuesSignal, m_writer, &ParameterWriter::setValues, Qt::QueuedConnection);
    m_writerThread.start();
}

Parameters::~Parameters()
{
    save();
    m_writerThread.quit();
    m_writerThread.wait();
}

Parameters &Parameters::Global()
{
    static Parameters params;
    return params;
}

void Parameters::load(const QString &path)
{
    QFile file(path);
    if (!file.exists())
    {
        file.open(QIODevice::Text | QIODevice::NewOnly);
        file.close();
    }
    m_settings.reset(new QSettings(path, QSettings::IniFormat));
    m_writer->setSettings(m_settings.data());

    m_cache.clear();
    QStringList keys = m_settings->allKeys();
    for (QStringList::iterator i = keys.begin(); i != keys.end(); i++)
    {
        qDebug() << *i << "--" << m_settings->value(*i);
        m_cache.insert(*i, m_settings->value(*i));
    }
}

QString Parameters::stringValue(const QString &key, const QString &defaultValue, const QString &group)
{
    QString fullKey = getFullKey(key, group);
    return m_cache.value(fullKey, defaultValue).toString();
}

bool Parameters::boolValue(const QString &key, const QString &defaultValue, const QString &group)
{
    QString fullKey = getFullKey(key, group);
    return m_cache.value(fullKey, defaultValue).toBool();
}

void Parameters::setValue(const QString& key, const QVariant& value, const QString &group)
{
    QString fullKey = getFullKey(key, group);
    m_cache[fullKey] = value;
    emit setValueSignal(key, value);
}

void Parameters::save()
{
    emit setValuesSignal(m_cache);
}

bool Parameters::debugMode()
{
    return boolValue("debug_mode");
}

void Parameters::setDebugMode(bool value)
{
    setValue("debug_mode", value);
}

QString Parameters::version()
{
    return stringValue("version");
}

void Parameters::setVersion(const QString &value)
{
    setValue("version", value);
}

QString Parameters::getFullKey(const QString& key, const QString& group)
{
    QString fullKey;
    if (group == DEFAULT_SETTINGS_GROUP)
    {
        fullKey = QString("%1/%2").arg(group).arg(key);
    }
    else
    {
        fullKey = key;
    }
    return fullKey;
}

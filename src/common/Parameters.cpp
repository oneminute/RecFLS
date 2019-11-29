#include "Parameters.h"

#include <QFile>

#define BEGIN_CHECK_GROUP(group) \
    if (group != DEFAULT_SETTINGS_GROUP) \
        m_settings->beginGroup(group)

#define CHECK_SETTING(key, defaultValue) \
    if (!m_settings->contains(key)) \
        setValue(key, defaultValue)

#define END_CHECK_GROUP(group) \
    if (group != DEFAULT_SETTINGS_GROUP) \
        m_settings->endGroup()

Parameters::Parameters(QObject *parent) : QObject(parent)
  , m_settings(nullptr)
{

}

Parameters::~Parameters()
{
    if (m_settings)
    {
        delete m_settings;
        m_settings = nullptr;
    }
}

Parameters &Parameters::Global()
{
    static Parameters params;
    return params;
}

void Parameters::load(const QString &path)
{
    if (m_settings)
    {
        delete m_settings;
    }

    QFile file(path);
    if (!file.exists())
    {
        file.open(QIODevice::Text | QIODevice::NewOnly);
        file.close();
    }
    m_settings = new QSettings(path, QSettings::IniFormat);
}

QString Parameters::stringValue(const QString &key, const QString &defaultValue, const QString &group)
{
    BEGIN_CHECK_GROUP(group);
    CHECK_SETTING(key, defaultValue);
    QString value = m_settings->value(key, defaultValue).toString();
    END_CHECK_GROUP(group);
    return value;
}

void Parameters::setValue(const QString &key, const QString &value, const QString &group)
{
    BEGIN_CHECK_GROUP(group);
    m_settings->setValue(key, value);
    END_CHECK_GROUP(group);
}

bool Parameters::boolValue(const QString &key, const QString &defaultValue, const QString &group)
{
    BEGIN_CHECK_GROUP(group);
    CHECK_SETTING(key, defaultValue);
    bool value = m_settings->value(key, defaultValue).toBool();
    END_CHECK_GROUP(group);
    return value;
}

void Parameters::setValue(const QString &key, bool value, const QString &group)
{
    BEGIN_CHECK_GROUP(group);
    m_settings->setValue(key, value);
    END_CHECK_GROUP(group);
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

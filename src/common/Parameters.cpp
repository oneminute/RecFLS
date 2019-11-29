#include "Parameters.h"

#include <QFile>

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
    if (group != DEFAULT_SETTINGS_GROUP)
        m_settings->beginGroup(group);
    if (!m_settings->contains(key))
        setValue(key, defaultValue);
    QString value = m_settings->value(key, defaultValue).toString();
    if (group != DEFAULT_SETTINGS_GROUP)
        m_settings->endGroup();
    return value;
}

void Parameters::setValue(const QString &key, const QString &value, const QString &group)
{
    if (group != DEFAULT_SETTINGS_GROUP)
        m_settings->beginGroup(group);
    m_settings->setValue(key, value);
    if (group != DEFAULT_SETTINGS_GROUP)
        m_settings->endGroup();
}

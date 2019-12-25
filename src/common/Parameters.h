#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <QObject>
#include <QSettings>

#define DEFAULT_SETTINGS_GROUP "General"

class Parameters : public QObject
{
    Q_OBJECT
public:
    explicit Parameters(QObject *parent = nullptr);

    ~Parameters();

    static Parameters& Global();

    void load(const QString &path = "config.ini");

    QString stringValue(const QString &key, const QString &defaultValue="", const QString &group = "General");
    void setValue(const QString &key, const QString &value, const QString &group = "General");

    bool boolValue(const QString &key, const QString &defaultValue="", const QString &group = "General");
    void setValue(const QString &key, bool value, const QString &group = "General");

    // [General] settings
    bool debugMode();
    void setDebugMode(bool value);

    QString version();
    void setVersion(const QString& value);
    // End [General] settings

signals:

public slots:

private:
    QSettings *m_settings;

};

#endif // PARAMETERS_H

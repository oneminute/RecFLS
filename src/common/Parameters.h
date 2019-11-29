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

signals:

public slots:

private:
    QSettings *m_settings;
};

#endif // PARAMETERS_H

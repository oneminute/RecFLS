#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <QObject>
#include <QElapsedTimer>
#include <QMultiMap>

#define TICK(key) StopWatch::instance().tick(key)
#define TOCK(key) StopWatch::instance().tock(key)

class StopWatch : public QObject
{
    Q_OBJECT
public:
    static StopWatch& instance();

    explicit StopWatch(QObject *parent = nullptr);

    void start();

    void tick(const QString& key);

    void tock(const QString& key);

    void debugPrint();

signals:

private:
    QElapsedTimer m_timer;
    QMultiMap<QString, QPair<qint64, qint64>> m_logs;
    QMap<QString, QPair<qint64, qint64>> m_pairs;
};

#endif // STOPWATCH_H

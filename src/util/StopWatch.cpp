#include "StopWatch.h"

#include <QDebug>
#include <QList>

StopWatch& StopWatch::instance()
{
    static StopWatch inst;
    return inst;
}

StopWatch::StopWatch(QObject *parent) : QObject(parent)
{
    m_timer.start();
}

void StopWatch::start()
{
    m_timer.restart();
}

void StopWatch::tick(const QString& key)
{
    qint64 stamp = m_timer.nsecsElapsed();
    QPair<qint64, qint64> pair(stamp, -1);
    m_pairs[key] = pair;
}

void StopWatch::tock(const QString& key)
{
    qint64 stamp = m_timer.nsecsElapsed();
    QPair<qint64, qint64> pair = m_pairs[key];
    pair.second = stamp;
    m_logs.insert(key, pair);
}

void StopWatch::debugPrint()
{
    QStringList keys = m_logs.keys();
    for (QStringList::iterator i = keys.begin(); i != keys.end(); i++)
    {
        QList<QPair<qint64, qint64>> records = m_logs.values(*i);
        qreal totalDuration = 0;
        int count = 0;
        for (int j = 0; j < records.size(); j++)
        {
            if (records[j].second >= records[j].first)
            {
                qreal duration = (records[j].second - records[j].first) / 1000000.0;
                totalDuration += duration;
                count++;
            }
        }
        qDebug().noquote() << *i << "duration:" << (totalDuration / count) << "ms";
    }
}

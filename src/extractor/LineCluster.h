#ifndef LINECLUSTER_H
#define LINECLUSTER_H

#include "LineSegment.h"

#include <QObject>

class LineCluster : public QObject
{
    Q_OBJECT
public:
    explicit LineCluster(QObject *parent = nullptr);

    void addLine(const LineSegment &line);

    QList<LineSegment> lines() const
    {
        return m_lines;
    }

    LineSegment merge();

    int size() const
    {
        return m_lines.size();
    }

private:
    QList<LineSegment> m_lines;
    float m_maxLength;
    float m_sumLength;
};

#endif // LINECLUSTER_H

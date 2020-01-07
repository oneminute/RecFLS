#include "LineCluster.h"
#include "util/Utils.h"

#include <Eigen/Core>
#include <Eigen/Dense>

LineCluster::LineCluster(QObject *parent) 
    : QObject(parent)
    , m_maxLength(0)
    , m_sumLength(0)
{

}

void LineCluster::addLine(LineSegment &line)
{
    if (!m_lines.isEmpty())
    {
        if (m_lines[0].direction().dot(line.direction()) < 0)
        {
            qDebug() << "reverse.";
            line.reverse();
        }
    }
    m_lines.append(line);
    m_sumLength += line.length();
}

LineSegment LineCluster::merge()
{
    QList<int> weights;
    Eigen::Vector3f middle(Eigen::Vector3f::Zero());
    Eigen::Vector3f dir(Eigen::Vector3f::Zero());

    for (int i = 0; i < m_lines.size(); i++)
    {
        float weight = m_lines[i].length() / m_sumLength;
        dir += m_lines[i].direction().normalized() * weight;
        middle += m_lines[i].middle() * weight;
    }
    dir = dir.normalized();
    //qDebug() << m_sumLength << middle.x() << middle.y() << middle.z() << dir.x() << dir.y() << dir.z();

    Eigen::Vector3f start = m_lines[0].start();
    Eigen::Vector3f end = m_lines[0].end();

    float startDistance = 0;
    float endDistance = 0;
    for (int i = 1; i < m_lines.size(); i++)
    {
        /*Eigen::Vector3f sm = middle - m_lines[i].start();
        Eigen::Vector3f me = m_lines[i].end() - middle;
        float sd = qAbs(sm.norm());
        float ed = qAbs(me.norm());
        if (sm.dot(dir) >= 0 && sd > startDistance)
        {
            startDistance = sd;
            start = m_lines[i].start();
        }

        if (me.dot(dir) >= 0 && ed > endDistance)
        {
            endDistance = ed;
            end = m_lines[i].end();
        }*/
        Eigen::Vector3f ss = start - m_lines[i].start();
        Eigen::Vector3f ee = m_lines[i].end() - end;

        if (ss.dot(dir) >= 0)
        {
            start = m_lines[i].start();
        }

        if (ee.dot(dir) >= 0)
        {
            end = m_lines[i].end();
        }
    }

    start = closedPointOnLine(start, dir, middle);
    end = closedPointOnLine(end, dir, middle);
    //start = closedPointOnLine(m_lines[0].start(), dir, middle);
    //end = closedPointOnLine(m_lines[m_lines.length() - 1].end(), dir, middle);

    LineSegment line(start, end);
    return line;
}

#include "LineCluster.h"
#include "util/Utils.h"

#include <Eigen/Core>
#include <Eigen/Dense>

LineCluster::LineCluster(QObject *parent) : QObject(parent)
{

}

void LineCluster::addLine(const LineSegment &line)
{
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

    Eigen::Vector3f start = closedPointOnLine(m_lines[0].start(), dir, middle);
    Eigen::Vector3f end = closedPointOnLine(m_lines[m_lines.size() - 1].end(), dir, middle);

    LineSegment line(start, end);
    return line;
}

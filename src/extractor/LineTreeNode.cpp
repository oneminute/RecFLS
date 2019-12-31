#include "LineTreeNode.h"

LineTreeNode::LineTreeNode(const LineSegment &line, QObject *parent)
    : QObject(parent)
    , m_line(line)
    , m_parent(nullptr)
    , m_leftChild(nullptr)
    , m_rightChild(nullptr)
    , m_distance(0)
    , m_chainDistance(0)
    , m_accessed(false)
{

}

LineTreeNode::~LineTreeNode()
{
    if (m_leftChild)
    {
        m_leftChild->deleteLater();
    }

    if (m_rightChild)
    {
        m_rightChild->deleteLater();
    }
}

void LineTreeNode::setLine(const LineSegment &line)
{
    m_line = line;
}

LineSegment LineTreeNode::line() const
{
    return m_line;
}

LineTreeNode *LineTreeNode::leftChild() const
{
    return m_leftChild;
}

LineTreeNode *LineTreeNode::rightChild() const
{
    return m_rightChild;
}

void LineTreeNode::addLeftChild(LineTreeNode *node)
{
    m_leftChild = node;
    node->setParent(this);
}

void LineTreeNode::addSideChild(LineTreeNode *node)
{
    m_sideLines.append(node);
    node->setParent(this);
}

void LineTreeNode::addRightChild(LineTreeNode *node)
{
    m_rightChild = node;
    if (node)
        node->setParent(this);
}

bool LineTreeNode::valid() const
{
    return m_line.available();
}

bool LineTreeNode::isLeaf() const
{
    return !hasLeftChild() && !hasRightChild();
}

bool LineTreeNode::hasParent() const
{
    return m_parent != nullptr;
}

bool LineTreeNode::hasLeftChild() const
{
    return m_leftChild != nullptr;
}

bool LineTreeNode::hasRightChild() const
{
    return m_rightChild != nullptr;
}

float LineTreeNode::distance() const
{
    return m_distance;
}

void LineTreeNode::setDistance(float distance)
{
    m_distance = distance;
}

float LineTreeNode::chainDistance() const
{
    return m_chainDistance;
}

void LineTreeNode::setChainDistance(float chainDistance)
{
    m_chainDistance = chainDistance;
}

LineTreeNode *LineTreeNode::parent() const
{
    return m_parent;
}

void LineTreeNode::setParent(LineTreeNode *parent)
{
    m_parent = parent;
}

QList<LineTreeNode *> &LineTreeNode::sideLines()
{
    return m_sideLines;
}

bool LineTreeNode::isRightRoot() const
{
    if (parent() == nullptr)
    {
        return true;
    }
    else if (this == parent()->rightChild())
    {
        return true;
    }
    return false;
}

bool LineTreeNode::isLeftChild() const
{
    return hasParent() && (parent()->leftChild() == this);
}

bool LineTreeNode::isRightChild() const
{
    return hasParent() && (parent()->rightChild() == this);
}

bool LineTreeNode::accessed() const
{
    return m_accessed;
}

void LineTreeNode::setAccessed(bool accessed)
{
    m_accessed = accessed;
}

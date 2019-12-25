#include "LineTreeNode.h"

LineTreeNode::LineTreeNode(const LineSegment &line, QObject *parent)
    : QObject(parent)
    , m_line(line)
    , m_chainChild(nullptr)
    , m_sideChild(nullptr)
    , m_farChild(nullptr)
    , m_rightChild(nullptr)
{

}

LineTreeNode::LineTreeNode(const LineTreeNode &rhs)
    : m_line(rhs.line())
    , m_chainChild(rhs.chainChild())
    , m_sideChild(rhs.sideChild())
    , m_farChild(rhs.farChild())
    , m_rightChild(rhs.rightChild())
{

}

LineTreeNode &LineTreeNode::operator=(const LineTreeNode &rhs)
{
    if (this != &rhs)
    {
        m_line = rhs.line();
        m_chainChild = rhs.chainChild();
        m_sideChild = rhs.sideChild();
        m_farChild = rhs.farChild();
        m_rightChild = rhs.rightChild();
    }
    return *this;
}

LineTreeNode::~LineTreeNode()
{
    if (m_chainChild)
    {
        m_chainChild->deleteLater();
    }

    if (m_sideChild)
    {
        m_sideChild->deleteLater();
    }

    if (m_farChild)
    {
        m_farChild->deleteLater();
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

LineTreeNode *LineTreeNode::chainChild() const
{
    return m_chainChild;
}

LineTreeNode *LineTreeNode::sideChild() const
{
    return m_sideChild;
}

LineTreeNode *LineTreeNode::farChild() const
{
    return m_farChild;
}

LineTreeNode *LineTreeNode::rightChild() const
{
    return m_rightChild;
}

void LineTreeNode::addChainChild(LineTreeNode *node)
{
    m_chainChild = node;
}

void LineTreeNode::addSideChild(LineTreeNode *node)
{
    m_sideChild = node;
}

void LineTreeNode::addFarChild(LineTreeNode *node)
{
    m_farChild = node;
}

void LineTreeNode::addRightChild(LineTreeNode *node)
{
    m_rightChild = node;
}

bool LineTreeNode::valid() const
{
    return m_line.available();
}

bool LineTreeNode::isLeaf() const
{
    return !hasSideChild() && !hasSideChild() && !hasFarChild() && !hasRightChild();
}

bool LineTreeNode::hasChainChild() const
{
    return m_chainChild != nullptr;
}

bool LineTreeNode::hasSideChild() const
{
    return m_sideChild != nullptr;
}

bool LineTreeNode::hasFarChild() const
{
    return m_farChild != nullptr;
}

bool LineTreeNode::hasRightChild() const
{
    return m_rightChild != nullptr;
}

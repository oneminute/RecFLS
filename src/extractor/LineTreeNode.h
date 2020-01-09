#ifndef LINETREENODE_H
#define LINETREENODE_H

#include <QObject>
#include <QSharedDataPointer>

#include "LineSegment.h"

class LineTreeNodeData;

class LineTreeNode : public QObject
{
    Q_OBJECT
public:
    explicit LineTreeNode(const LineSegment &line = LineSegment(), QObject *parent = nullptr);
    ~LineTreeNode();

    void setLine(const LineSegment &line);

    LineSegment line() const;

    LineTreeNode *leftChild() const;

    LineTreeNode *rightChild() const;

    void addLeftChild(LineTreeNode *node);

    void addSideChild(LineTreeNode *node);

    void addRightChild(LineTreeNode *node);

    bool valid() const;

    bool isLeaf() const;

    bool hasParent() const;

    bool hasLeftChild() const;

    bool hasRightChild() const;

    float distance() const;
    void setDistance(float distance);

    float chainDistance() const;
    void setChainDistance(float chainDistance);

    LineTreeNode *parent() const;
    void setParent(LineTreeNode *parent);

    QList<LineTreeNode *>& sideLines();

    bool isRightRoot() const;

    bool isLeftChild() const;

    bool isRightChild() const;

    bool accessed() const;
    void setAccessed(bool accessed = true);

private:
    LineSegment m_line;
    LineTreeNode *m_parent;
    LineTreeNode *m_leftChild;
    LineTreeNode *m_rightChild;
    float m_distance;
    float m_chainDistance;
    QList<LineTreeNode*> m_sideLines;
    bool m_accessed;
};

#endif // LINETREENODE_H

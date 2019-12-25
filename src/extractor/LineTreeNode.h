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
    LineTreeNode(const LineTreeNode &rhs);
    LineTreeNode &operator=(const LineTreeNode &rhs);
    ~LineTreeNode();

    void setLine(const LineSegment &line);

    LineSegment line() const;

    LineTreeNode *chainChild() const;

    LineTreeNode *sideChild() const;

    LineTreeNode *farChild() const;

    LineTreeNode *rightChild() const;

    void addChainChild(LineTreeNode *node);

    void addSideChild(LineTreeNode *node);

    void addFarChild(LineTreeNode *node);

    void addRightChild(LineTreeNode *node);

    bool valid() const;

    bool isLeaf() const;

    bool hasChainChild() const;

    bool hasSideChild() const;

    bool hasFarChild() const;

    bool hasRightChild() const;

private:
    LineSegment m_line;
    LineTreeNode *m_chainChild;
    LineTreeNode *m_sideChild;
    LineTreeNode *m_farChild;
    LineTreeNode *m_rightChild;
};

#endif // LINETREENODE_H
